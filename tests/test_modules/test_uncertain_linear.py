"""Test the class UncertainLinear"""
from inspect import signature

from hypothesis import given, strategies as hst
from numpy.testing import assert_equal
from torch import Tensor
from torch.nn import Module
from torch.testing import assert_close  # type: ignore[attr-defined]

from gum_compliant_neural_network_uncertainty_propagation import modules
from gum_compliant_neural_network_uncertainty_propagation.modules import (
    UncertainLinear,
)
from .conftest import (
    uncertain_linears,
    values_uncertainties_and_uncertain_linears,
    values_with_uncertainties,
    ValuesUncertainties,
    ValuesUncertaintiesForLinear,
)


def test_modules_all_contains_uncertain_linear() -> None:
    assert UncertainLinear.__name__ in modules.__all__


def test_modules_actually_contains_uncertain_linear() -> None:
    assert hasattr(modules, UncertainLinear.__name__)


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


def test_uncertain_linear_forward_accepts_two_parameters() -> None:
    assert_equal(len(signature(UncertainLinear.forward).parameters), 3)


def test_uncertain_linear_forward_accepts_values_parameter() -> None:
    assert "values" in signature(UncertainLinear.forward).parameters


def test_uncertain_linear_forward_accepts_uncertainties_parameter() -> None:
    assert "uncertainties" in signature(UncertainLinear.forward).parameters


def test_uncertain_linear_forward_expects_values_parameter_as_tensor() -> None:
    assert issubclass(
        signature(UncertainLinear.forward).parameters["values"].annotation, Tensor
    )


@given(values_with_uncertainties(), hst.integers(min_value=1, max_value=10))
def test_uncertain_linear_forward_returns_tuple(
    values_and_uncertainties: ValuesUncertainties, out_features: int
) -> None:
    assert isinstance(
        UncertainLinear(len(values_and_uncertainties.values), out_features).forward(
            values_and_uncertainties.values.float(),
            values_and_uncertainties.uncertainties.float(),
        ),
        tuple,
    )


@given(values_uncertainties_and_uncertain_linears())
def test_uncertain_linear_forward_returns_tensors(
    values_uncertainties_and_uncertain_linear: ValuesUncertaintiesForLinear,
) -> None:
    for (
        return_element
    ) in values_uncertainties_and_uncertain_linear.uncertain_linear.forward(
        values_uncertainties_and_uncertain_linear.values.float(),
        values_uncertainties_and_uncertain_linear.uncertainties.float(),
    ):
        assert isinstance(return_element, Tensor)


@given(values_uncertainties_and_uncertain_linears())
def test_default_uncertain_linear_forward_provides_no_uncertainties_if_not_provided(
    values_uncertainties_and_uncertain_linear: ValuesUncertaintiesForLinear,
) -> None:
    assert (
        values_uncertainties_and_uncertain_linear.uncertain_linear.forward(
            values_uncertainties_and_uncertain_linear.values.float(), None
        )[1]
        is None
    )


@given(values_uncertainties_and_uncertain_linears())
def test_uncertain_linear_forward_uncertainties_for_random_input(
    values_uncertainties_and_uncertain_linear: ValuesUncertaintiesForLinear,
) -> None:
    assert_close(
        values_uncertainties_and_uncertain_linear.uncertain_linear.forward(
            values_uncertainties_and_uncertain_linear.values,
            values_uncertainties_and_uncertain_linear.uncertainties,
        )[1],
        values_uncertainties_and_uncertain_linear.uncertain_linear._linear.weight
        @ values_uncertainties_and_uncertain_linear.uncertainties
        @ values_uncertainties_and_uncertain_linear.uncertain_linear._linear.weight.T,
    )


@given(values_uncertainties_and_uncertain_linears())
def test_uncertain_linear_forward_values_for_random_input(
    values_uncertainties_and_uncertain_linear: ValuesUncertaintiesForLinear,
) -> None:
    assert_close(
        values_uncertainties_and_uncertain_linear.uncertain_linear.forward(
            values_uncertainties_and_uncertain_linear.values,
            values_uncertainties_and_uncertain_linear.uncertainties,
        )[0],
        (
            values_uncertainties_and_uncertain_linear.uncertain_linear._linear.forward(
                values_uncertainties_and_uncertain_linear.values
            )
        ),
    )


@given(values_uncertainties_and_uncertain_linears())
def test_uncertain_linear_forward_for_random_input(
    values_uncertainties_and_uncertain_linear: ValuesUncertaintiesForLinear,
) -> None:
    linear_layer = values_uncertainties_and_uncertain_linear.uncertain_linear._linear
    assert_close(
        values_uncertainties_and_uncertain_linear.uncertain_linear.forward(
            values_uncertainties_and_uncertain_linear.values,
            values_uncertainties_and_uncertain_linear.uncertainties,
        ),
        (
            linear_layer.forward(values_uncertainties_and_uncertain_linear.values),
            linear_layer.weight
            @ values_uncertainties_and_uncertain_linear.uncertainties
            @ linear_layer.weight.T,
        ),
    )
