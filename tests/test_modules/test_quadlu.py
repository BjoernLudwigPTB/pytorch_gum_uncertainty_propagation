"""Test the class QuadLU"""
from inspect import signature

from hypothesis import given
from numpy.testing import assert_equal
from torch import equal, square, Tensor
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.testing import assert_close  # type: ignore[attr-defined]

from pytorch_gum_uncertainty_propagation import modules
from pytorch_gum_uncertainty_propagation.modules import (
    QuadLU,
)
from ..conftest import alphas, tensors


def test_modules_all_contains_quadlu() -> None:
    assert QuadLU.__name__ in modules.__all__


def test_modules_actually_contains_quadlu() -> None:
    assert hasattr(modules, QuadLU.__name__)


def test_quadlu_is_subclass_of_nn_module() -> None:
    assert issubclass(QuadLU, Module)


def test_quadlu_has_docstring() -> None:
    assert QuadLU.__doc__ is not None


def test_init_quadlu(quadlu_instance: QuadLU) -> None:
    assert quadlu_instance


def test_quadlu_has_parameter_alpha() -> None:
    assert "alpha" in signature(QuadLU).parameters


def test_init_quadlu_creates_parameter(quadlu_instance: QuadLU) -> None:
    assert hasattr(quadlu_instance, "_alpha")


def test_quadlu_has_parameter_inplace() -> None:
    assert "inplace" in signature(QuadLU).parameters


def test_init_quadlu_creates_attribute_inplace(quadlu_instance: QuadLU) -> None:
    assert hasattr(quadlu_instance, "_inplace")


def test_init_quadlu_alpha_requires_grad(quadlu_instance: QuadLU) -> None:
    assert quadlu_instance._alpha.requires_grad


def test_init_quadlu_contains_constant_for_alphas_default(
    quadlu_instance: QuadLU,
) -> None:
    assert hasattr(quadlu_instance, "QUADLU_ALPHA_DEFAULT")


def test_init_quadlu_constant_for_alphas_default_value(quadlu_instance: QuadLU) -> None:
    assert_equal(quadlu_instance.QUADLU_ALPHA_DEFAULT.data.item(), 0.25)


def test_init_quadlu_creates_alpha_equal_to_default(quadlu_instance: QuadLU) -> None:
    assert equal(quadlu_instance._alpha, quadlu_instance.QUADLU_ALPHA_DEFAULT)


@given(alphas())
def test_init_quadlu_creates_parameter_alpha(alpha: Parameter) -> None:
    assert hasattr(QuadLU(alpha), "_alpha")


@given(alphas())
def test_init_quadlu_with_random_alpha(alpha: Parameter) -> None:
    assert_close(QuadLU(alpha)._alpha, alpha)


def test_quadlu_contains_callable_forward() -> None:
    assert callable(QuadLU.forward)


@given(tensors(elements_max=-QuadLU.QUADLU_ALPHA_DEFAULT.data.item()))
def test_default_quadlu_forward_for_small_x(x: Tensor) -> None:
    assert_equal(QuadLU().forward(x).data.numpy(), 0.0)


@given(tensors(elements_max=-0.16), alphas(max_value=0.16))
def test_quadlu_forward_for_small_x(x: Tensor, alpha: Parameter) -> None:
    assert_equal(QuadLU(alpha).forward(x).data.numpy(), 0.0)


@given(tensors(elements_min=QuadLU.QUADLU_ALPHA_DEFAULT.data.item()))
def test_default_quadlu_forward_for_large_x(x: Tensor) -> None:
    assert_close(QuadLU().forward(x), x)


@given(tensors(elements_min=0.15), alphas(max_value=0.15))
def test_quadlu_forward_large_x(x: Tensor, alpha: Parameter) -> None:
    assert_close(QuadLU(alpha).forward(x), 4 * alpha * x, equal_nan=True)


@given(
    tensors(
        elements_min=-QuadLU.QUADLU_ALPHA_DEFAULT.data.item(),
        elements_max=QuadLU.QUADLU_ALPHA_DEFAULT.data.item(),
    )
)
def test_default_quadlu_forward_near_zero(x: Tensor) -> None:
    assert_close(QuadLU().forward(x), square(x + QuadLU.QUADLU_ALPHA_DEFAULT))


@given(tensors(elements_min=-0.14, elements_max=0.14), alphas(min_value=0.14))
def test_quadlu_forward_near_zero(x: Tensor, alpha: Parameter) -> None:
    assert_close(QuadLU(alpha).forward(x), square(x + alpha))


@given(tensors())
def test_default_quadlu_forward_for_random_input(values: Tensor) -> None:
    less_or_equal_mask = values <= -QuadLU.QUADLU_ALPHA_DEFAULT
    greater_or_equal_mask = values >= QuadLU.QUADLU_ALPHA_DEFAULT
    in_between_mask = ~(less_or_equal_mask | greater_or_equal_mask)
    result_tensor = QuadLU().forward(values)
    assert_equal(result_tensor[less_or_equal_mask].data.numpy(), 0.0)
    assert_close(
        result_tensor[in_between_mask],
        square(values[in_between_mask] + QuadLU.QUADLU_ALPHA_DEFAULT),
        equal_nan=True,
    )
    assert_close(result_tensor[greater_or_equal_mask], values[greater_or_equal_mask])


@given(tensors(), alphas())
def test_quadlu_forward_for_random_input(values: Tensor, alpha: Parameter) -> None:
    less_or_equal_mask = values <= -alpha
    greater_or_equal_mask = values >= alpha
    in_between_mask = ~(less_or_equal_mask | greater_or_equal_mask)
    result_tensor = QuadLU(alpha).forward(values)
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
def test_default_quadlu_inplace_is_inplace(values: Tensor) -> None:
    assert_close(QuadLU(inplace=True).forward(values), values, equal_nan=True)


@given(tensors(), alphas())
def test_quadlu_inplace_is_inplace(values: Tensor, alpha: Parameter) -> None:
    assert_close(QuadLU(alpha, inplace=True).forward(values), values, equal_nan=True)


@given(tensors())
def test_default_inplace_quadlu_equals_quadlu(values: Tensor) -> None:
    assert_close(
        QuadLU().forward(values), QuadLU(inplace=True).forward(values), equal_nan=True
    )


@given(tensors(), alphas())
def test_inplace_quadlu_equals_quadlu(values: Tensor, alpha: Parameter) -> None:
    assert_close(
        QuadLU(alpha).forward(values),
        QuadLU(alpha, inplace=True).forward(values),
        equal_nan=True,
    )
