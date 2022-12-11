"""Test the class UncertainQuadLU"""
from inspect import signature
from typing import cast

from hypothesis import given, settings
from hypothesis.strategies import composite, DrawFn, SearchStrategy
from numpy.testing import assert_equal
from torch import square, Tensor
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.testing import assert_close  # type: ignore[attr-defined]

from gum_compliant_neural_network_uncertainty_propagation import modules
from gum_compliant_neural_network_uncertainty_propagation.modules import (
    QuadLU,
    UncertainQuadLU,
)
from gum_compliant_neural_network_uncertainty_propagation.uncertainties import (
    cov_matrix_from_std_uncertainties,
)
from ..conftest import alphas, tensors


@composite
def values_with_uncertainties(draw: DrawFn) -> dict[str, Tensor]:
    values: Tensor = cast(Tensor, draw(tensors(elements_min=1e2, elements_max=1e4)))
    std_uncertainties = cast(
        Tensor, draw(tensors(elements_min=0.1, elements_max=10, length=len(values)))
    )
    cov_matrix = cov_matrix_from_std_uncertainties(std_uncertainties, 0.5, 0.5, 0.5)
    return {"values": values, "std_uncertainties": cov_matrix}


def test_modules_all_contains_uncertain_quadlu() -> None:
    assert UncertainQuadLU.__name__ in modules.__all__


def test_modules_actually_contains_uncertain_quadlu() -> None:
    assert hasattr(modules, UncertainQuadLU.__name__)


def test_uncertain_quadlu_is_subclass_of_nn_module() -> None:
    assert issubclass(UncertainQuadLU, Module)


def test_uncertain_quadlu_has_docstring() -> None:
    assert UncertainQuadLU.__doc__ is not None


def test_init_uncertain_quadlu(uncertain_quadlu_instance: UncertainQuadLU) -> None:
    assert uncertain_quadlu_instance


def test_uncertain_quadlu_has_parameter_alpha() -> None:
    assert "alpha" in signature(UncertainQuadLU).parameters


def test_init_uncertain_quadlu_creates_attribute_alpha(
    uncertain_quadlu_instance: UncertainQuadLU,
) -> None:
    assert hasattr(uncertain_quadlu_instance, "_alpha")


def test_uncertain_quadlu_has_parameter_inplace() -> None:
    assert "inplace" in signature(UncertainQuadLU).parameters


def test_init_uncertain_quadlu_creates_attribute_inplace(
    uncertain_quadlu_instance: UncertainQuadLU,
) -> None:
    assert hasattr(uncertain_quadlu_instance, "_inplace")


def test_init_uncertain_quadlu_creates_attribute_quadlu(
    uncertain_quadlu_instance: UncertainQuadLU,
) -> None:
    assert hasattr(uncertain_quadlu_instance, "_quadlu")


def test_init_uncertain_quadlu_alpha_requires_grad(
    uncertain_quadlu_instance: UncertainQuadLU,
) -> None:
    assert uncertain_quadlu_instance._alpha.requires_grad


def test_init_uncertain_quadlu_contains_constant_for_alphas_default(
    uncertain_quadlu_instance: UncertainQuadLU,
) -> None:
    assert hasattr(uncertain_quadlu_instance, "QUADLU_ALPHA_DEFAULT")


def test_init_default_uncertain_quadlu_and_quadlus_creates_same_alpha(
    uncertain_quadlu_instance: UncertainQuadLU, quadlu_instance: QuadLU
) -> None:
    assert_close(
        uncertain_quadlu_instance.QUADLU_ALPHA_DEFAULT,
        quadlu_instance.QUADLU_ALPHA_DEFAULT,
    )


def test_init_uncertain_quadlu_creates_alpha_equal_to_default(
    uncertain_quadlu_instance: UncertainQuadLU,
) -> None:
    assert_close(
        uncertain_quadlu_instance._alpha, uncertain_quadlu_instance.QUADLU_ALPHA_DEFAULT
    )


@given(alphas())
def test_init_uncertain_quadlu_creates_parameter_alpha(alpha: Parameter) -> None:
    assert hasattr(UncertainQuadLU(alpha), "_alpha")


@given(alphas())
def test_init_uncertain_quadlu_with_random_alpha(alpha: Parameter) -> None:
    assert_close(UncertainQuadLU(alpha)._alpha, alpha)


def test_uncertain_quadlu_contains_callable_forward() -> None:
    assert callable(UncertainQuadLU.forward)


@given(tensors(elements_max=-UncertainQuadLU.QUADLU_ALPHA_DEFAULT.data.item()))
def test_default_uncertain_quadlu_forward_for_small_x(x: Tensor) -> None:
    assert_equal(UncertainQuadLU().forward(x)[0].data.numpy(), 0.0)


@given(tensors(elements_max=-0.1501), alphas(max_value=0.15))
def test_uncertain_quadlu_forward_for_small_x(x: Tensor, alpha: Parameter) -> None:
    assert_equal(UncertainQuadLU(alpha).forward(x)[0].data.numpy(), 0.0)


@given(tensors())
def test_default_uncertain_quadlu_forward_provides_no_uncertainties_if_not_provided(
    x: Tensor,
) -> None:
    assert UncertainQuadLU().forward(x)[1] is None


@given(tensors(elements_min=UncertainQuadLU.QUADLU_ALPHA_DEFAULT.data.item()))
def test_default_uncertain_quadlu_forward_for_large_x(x: Tensor) -> None:
    assert_close(UncertainQuadLU().forward(x)[0], x)


@given(tensors(elements_min=0.15), alphas(max_value=0.15))
def test_uncertain_quadlu_forward_for_large_x(x: Tensor, alpha: Parameter) -> None:
    assert_close(UncertainQuadLU(alpha).forward(x)[0], 4 * alpha * x, equal_nan=True)


@given(
    tensors(
        elements_min=-UncertainQuadLU.QUADLU_ALPHA_DEFAULT.data.item(),
        elements_max=UncertainQuadLU.QUADLU_ALPHA_DEFAULT.data.item(),
    )
)
def test_default_uncertain_quadlu_forward_near_zero(x: Tensor) -> None:
    assert_close(
        UncertainQuadLU().forward(x)[0],
        square(x + UncertainQuadLU.QUADLU_ALPHA_DEFAULT),
    )


@given(tensors(elements_min=-0.14, elements_max=0.14), alphas(min_value=0.14))
def test_uncertain_quadlu_forward_near_zero(x: Tensor, alpha: Parameter) -> None:
    assert_close(UncertainQuadLU(alpha).forward(x)[0], square(x + alpha))


@given(tensors())
def test_default_uncertain_quadlu_forward_for_random_input(values: Tensor) -> None:
    less_or_equal_mask = values <= -UncertainQuadLU.QUADLU_ALPHA_DEFAULT
    greater_or_equal_mask = values >= UncertainQuadLU.QUADLU_ALPHA_DEFAULT
    in_between_mask = ~(less_or_equal_mask | greater_or_equal_mask)
    result_tensor = UncertainQuadLU().forward(values)[0]
    assert_equal(result_tensor[less_or_equal_mask].data.numpy(), 0.0)
    assert_close(
        result_tensor[in_between_mask],
        square(values[in_between_mask] + UncertainQuadLU.QUADLU_ALPHA_DEFAULT),
        equal_nan=True,
    )
    assert_close(result_tensor[greater_or_equal_mask], values[greater_or_equal_mask])


@given(tensors(), alphas())
def test_uncertain_quadlu_forward_for_random_input(
    values: Tensor, alpha: Parameter
) -> None:
    less_or_equal_mask = values <= -alpha
    greater_or_equal_mask = values >= alpha
    in_between_mask = ~(less_or_equal_mask | greater_or_equal_mask)
    result_tensor = UncertainQuadLU(alpha).forward(values)[0]
    assert_equal(result_tensor[less_or_equal_mask].data.numpy(), 0.0)
    assert_close(
        result_tensor[in_between_mask],
        square(values[in_between_mask] + alpha),
        equal_nan=True,
    )
    assert_close(
        result_tensor[greater_or_equal_mask],
        4 * alpha * values[greater_or_equal_mask],
        equal_nan=True,
    )


@given(tensors())
def test_default_uncertain_quadlu_inplace_is_inplace(values: Tensor) -> None:
    assert_close(
        UncertainQuadLU(inplace=True).forward(values)[0], values, equal_nan=True
    )


@given(tensors(), alphas())
def test_uncertain_quadlu_inplace_is_inplace(values: Tensor, alpha: Parameter) -> None:
    assert_close(
        UncertainQuadLU(alpha, inplace=True).forward(values)[0], values, equal_nan=True
    )


@given(tensors())
def test_default_inplace_uncertain_quadlu_equals_quadlu(values: Tensor) -> None:
    assert_close(
        UncertainQuadLU().forward(values),
        UncertainQuadLU(inplace=True).forward(values),
        equal_nan=True,
    )


@given(tensors(), alphas())
def test_inplace_uncertain_quadlu_equals_quadlu(
    values: Tensor, alpha: Parameter
) -> None:
    assert_close(
        UncertainQuadLU(alpha).forward(values),
        UncertainQuadLU(alpha, inplace=True).forward(values),
        equal_nan=True,
    )


@given(values_with_uncertainties())
@settings(deadline=None)
def test_default_uncertain_quadlu_forward_accepts_random_input(
    values_and_uncertainties: dict[str, Tensor]
) -> None:
    assert UncertainQuadLU().forward(
        values_and_uncertainties["values"],
        values_and_uncertainties["std_uncertainties"],
    )
