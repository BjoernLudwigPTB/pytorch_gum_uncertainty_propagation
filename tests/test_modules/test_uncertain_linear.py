"""Test the class UncertainLinear"""
from inspect import signature

from hypothesis import given, settings, strategies as hst
from numpy.testing import assert_equal
from torch import Tensor
from torch.nn import Linear, Module
from torch.testing import assert_close  # type: ignore[attr-defined]

from pytorch_gum_uncertainty_propagation import modules
from pytorch_gum_uncertainty_propagation.modules import UncertainLinear
from pytorch_gum_uncertainty_propagation.uncertainties import (
    _is_positive_semi_definite,
    _is_symmetric,
    UncertainTensor,
)
from .conftest import (
    uncertain_linears,
    UncertainTensorForLinear,
    values_uncertainties_and_uncertain_linears,
)
from ..conftest import uncertain_tensors


def test_modules_actually_contains_uncertain_linear() -> None:
    assert hasattr(modules, UncertainLinear.__name__)


def test_modules_all_contains_uncertain_linear() -> None:
    assert UncertainLinear.__name__ in modules.__all__


def test_uncertain_linear_is_subclass_of_nn_module() -> None:
    assert issubclass(UncertainLinear, Module)


def test_uncertain_linear_has_docstring() -> None:
    assert UncertainLinear.__doc__ is not None


@given(uncertain_linears())
def test_init_uncertain_linear(uncertain_linear_instance: UncertainLinear) -> None:
    assert uncertain_linear_instance


def test_uncertain_linear_has_parameter_in_features() -> None:
    assert "in_features" in signature(UncertainLinear).parameters


def test_uncertain_linear_has_parameter_out_features() -> None:
    assert "out_features" in signature(UncertainLinear).parameters


def test_uncertain_linear_has_parameter_bias() -> None:
    assert "bias" in signature(UncertainLinear).parameters


@given(uncertain_linears())
def test_init_uncertain_linear_creates_attribute_in_features(
    uncertain_linear_instance: UncertainLinear,
) -> None:
    assert hasattr(uncertain_linear_instance, "in_features")


def test_uncertain_quadlu_in_features_has_docstring() -> None:
    assert UncertainLinear.in_features.__doc__ is not None


@given(uncertain_linears())
def test_init_uncertain_linear_creates_attribute_out_features(
    uncertain_linear_instance: UncertainLinear,
) -> None:
    assert hasattr(uncertain_linear_instance, "out_features")


def test_uncertain_quadlu_out_features_has_docstring() -> None:
    assert UncertainLinear.out_features.__doc__ is not None


@given(uncertain_linears())
def test_init_uncertain_linear_creates_attribute_bias(
    uncertain_linear_instance: UncertainLinear,
) -> None:
    assert hasattr(uncertain_linear_instance, "bias")


def test_uncertain_quadlu_bias_has_docstring() -> None:
    assert UncertainLinear.bias.__doc__ is not None


@given(uncertain_linears())
def test_init_uncertain_linear_creates_attribute_weight(
    uncertain_linear_instance: UncertainLinear,
) -> None:
    assert hasattr(uncertain_linear_instance, "weight")


def test_uncertain_quadlu_weight_has_docstring() -> None:
    assert UncertainLinear.weight.__doc__ is not None


@given(uncertain_linears(in_features=5))
def test_init_uncertain_linear_creates_attribute_in_features_correctly(
    uncertain_linear_instance: UncertainLinear,
) -> None:
    assert_equal(uncertain_linear_instance.in_features, 5)


@given(uncertain_linears())
def test_init_uncertain_linear_creates_default_attribute_bias_true(
    uncertain_linear_instance: UncertainLinear,
) -> None:
    assert isinstance(uncertain_linear_instance.bias, Tensor)


@given(uncertain_linears(bias=False))
def test_init_uncertain_linear_creates_attribute_bias_false(
    uncertain_linear_instance: UncertainLinear,
) -> None:
    assert uncertain_linear_instance.bias is None


def test_uncertain_linear_contains_callable_forward() -> None:
    assert callable(UncertainLinear.forward)


def test_uncertain_linear_forward_accepts_one_parameters() -> None:
    assert_equal(len(signature(UncertainLinear.forward).parameters), 2)


def test_uncertain_linear_forward_accepts_uncertain_values_parameter() -> None:
    assert "uncertain_values" in signature(UncertainLinear.forward).parameters


def test_uncertain_linear_forward_expects_uncertain_tensor_parameter() -> None:
    assert issubclass(
        signature(UncertainLinear.forward).parameters["uncertain_values"].annotation,
        UncertainTensor,
    )


@given(uncertain_tensors(), hst.integers(min_value=1, max_value=10))
def test_uncertain_linear_forward_returns_uncertain_tensor(
    values_and_uncertainties: UncertainTensor, out_features: int
) -> None:
    assert isinstance(
        UncertainLinear(len(values_and_uncertainties.values), out_features).forward(
            values_and_uncertainties,
        ),
        UncertainTensor,
    )


@given(values_uncertainties_and_uncertain_linears())
def test_uncertain_linear_forward_returns_tensors(
    values_uncertainties_and_uncertain_linear: UncertainTensorForLinear,
) -> None:
    uncertain_result = (
        values_uncertainties_and_uncertain_linear.uncertain_linear.forward(
            values_uncertainties_and_uncertain_linear.uncertain_values
        )
    )
    assert uncertain_result.uncertainties is not None
    for return_element in uncertain_result:
        assert isinstance(return_element, Tensor)


@given(values_uncertainties_and_uncertain_linears())
def test_default_uncertain_linear_forward_provides_no_uncertainties_if_not_provided(
    values_uncertainties_and_uncertain_linear: UncertainTensorForLinear,
) -> None:
    certain_values = UncertainTensor(
        values_uncertainties_and_uncertain_linear.uncertain_values.values, None
    )
    assert (
        values_uncertainties_and_uncertain_linear.uncertain_linear.forward(
            certain_values
        ).uncertainties
        is None
    )


@given(values_uncertainties_and_uncertain_linears())
def test_uncertain_linear_forward_uncertainties_for_random_input(
    values_uncertainties_and_uncertain_linear: UncertainTensorForLinear,
) -> None:
    uncertain_linear = (
        values_uncertainties_and_uncertain_linear.uncertain_linear._linear
    )
    result_uncertainties = values_uncertainties_and_uncertain_linear.uncertain_linear(
        values_uncertainties_and_uncertain_linear.uncertain_values
    ).uncertainties
    assert result_uncertainties is not None
    assert_close(
        result_uncertainties,
        uncertain_linear.weight
        @ values_uncertainties_and_uncertain_linear.uncertain_values.uncertainties
        @ uncertain_linear.weight.T,
    )


@given(values_uncertainties_and_uncertain_linears())
def test_uncertain_linear_forward_values_for_random_input(
    values_uncertainties_and_uncertain_linear: UncertainTensorForLinear,
) -> None:
    assert_close(
        values_uncertainties_and_uncertain_linear.uncertain_linear.forward(
            values_uncertainties_and_uncertain_linear.uncertain_values
        ).values,
        (
            values_uncertainties_and_uncertain_linear.uncertain_linear._linear(
                values_uncertainties_and_uncertain_linear.uncertain_values.values
            )
        ),
    )


@given(values_uncertainties_and_uncertain_linears())
def test_uncertain_linear_forward_for_random_input(
    uncertain_values_and_uncertain_linear: UncertainTensorForLinear,
) -> None:
    assert (
        uncertain_values_and_uncertain_linear.uncertain_values.uncertainties is not None
    )
    uncertain_linear = uncertain_values_and_uncertain_linear.uncertain_linear
    similar_linear = Linear(uncertain_linear.in_features, uncertain_linear.out_features)
    similar_linear.load_state_dict(uncertain_linear._linear.state_dict())
    assert_close(
        uncertain_linear.forward(
            uncertain_values_and_uncertain_linear.uncertain_values
        ),
        (
            similar_linear(
                uncertain_values_and_uncertain_linear.uncertain_values.values
            ),
            uncertain_linear.weight
            @ uncertain_values_and_uncertain_linear.uncertain_values.uncertainties
            @ uncertain_linear.weight.T,
        ),
    )


@given(values_uncertainties_and_uncertain_linears())
@settings(deadline=None)
def test_uncertain_linear_forward_results_in_positive_semi_definite_uncertainties(
    uncertain_values_and_uncertain_linear: UncertainTensorForLinear,
) -> None:
    uncertain_linear = uncertain_values_and_uncertain_linear.uncertain_linear
    result_uncertainties = uncertain_linear.forward(
        uncertain_values_and_uncertain_linear.uncertain_values
    ).uncertainties
    assert result_uncertainties is not None
    assert _is_positive_semi_definite(result_uncertainties)


@given(values_uncertainties_and_uncertain_linears())
def test_uncertain_linear_forward_results_in_symmetric_uncertainties(
    uncertain_values_and_uncertain_linear: UncertainTensorForLinear,
) -> None:
    uncertain_linear = uncertain_values_and_uncertain_linear.uncertain_linear
    result_uncertainties = uncertain_linear.forward(
        uncertain_values_and_uncertain_linear.uncertain_values
    ).uncertainties
    assert result_uncertainties is not None
    assert _is_symmetric(result_uncertainties)
