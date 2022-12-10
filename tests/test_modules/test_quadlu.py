"""Test the class QuadLU"""
from hypothesis import given
from numpy.testing import assert_equal
from torch import equal, tensor, Tensor
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.testing import assert_close  # type: ignore[attr-defined]

from gum_compliant_neural_network_uncertainty_propagation import modules
from gum_compliant_neural_network_uncertainty_propagation.modules import (
    QuadLU,
)
from ..conftest import alphas, tensors


def test_modules_all_contains_quadlu() -> None:
    assert QuadLU.__name__ in modules.__all__


def test_quadlu_is_subclass_of_nn_module() -> None:
    assert issubclass(QuadLU, Module)


def test_quadlu_has_docstring() -> None:
    assert QuadLU.__doc__ is not None


def test_init_quadlu(quadlu: QuadLU) -> None:
    assert quadlu


def test_init_quadlu_creates_parameter(quadlu: QuadLU) -> None:
    assert hasattr(quadlu, "_alpha")


def test_init_quadlu_alpha_requires_grad(quadlu: QuadLU) -> None:
    assert quadlu._alpha.requires_grad


def test_init_quadlu_contains_constant_for_alphas_default(quadlu: QuadLU) -> None:
    assert hasattr(quadlu, "QUADLU_ALPHA_DEFAULT")


def test_init_quadlu_constant_for_alphas_default_value(quadlu: QuadLU) -> None:
    assert_equal(quadlu.QUADLU_ALPHA_DEFAULT.data.item(), 0.25)


def test_init_quadlu_creates_alpha_equal_to_default(quadlu: QuadLU) -> None:
    assert equal(quadlu._alpha, quadlu.QUADLU_ALPHA_DEFAULT)


@given(alphas())
def test_init_quadlu_creates_parameter_with_custom_value(alpha: Parameter) -> None:
    assert hasattr(QuadLU(alpha), "_alpha")


@given(alphas())
def test_init_quadlu_with_random_alpha(alpha: Parameter) -> None:
    assert_close(QuadLU(alpha)._alpha, alpha)


def test_init_quadlu_contains_custom_forward(quadlu: QuadLU) -> None:
    assert quadlu.forward(tensor([1.0]))


@given(tensors(elements_max=-QuadLU.QUADLU_ALPHA_DEFAULT.data.item()))
def test_default_quadlu_forward_is_correct_for_small_x(x: Tensor) -> None:
    assert_equal(QuadLU().forward(x).data.numpy(), 0.0)


@given(tensors(elements_max=-0.16), alphas(max_value=0.16))
def test_quadlu_forward_is_correct_for_small_x(x: Tensor, alpha: Parameter) -> None:
    assert_equal(QuadLU(alpha).forward(x).data.numpy(), 0.0)


@given(tensors(elements_min=QuadLU.QUADLU_ALPHA_DEFAULT.data.item()))
def test_default_quadlu_forward_is_correct_for_large_x(x: Tensor) -> None:
    assert_close(QuadLU().forward(x), x, equal_nan=True)


@given(tensors(elements_min=0.15), alphas(max_value=0.15))
def test_quadlu_forward_is_correct_large_x(x: Tensor, alpha: Parameter) -> None:
    assert_close(QuadLU(alpha).forward(x), 4 * alpha * x, equal_nan=True)


@given(
    tensors(
        elements_min=-QuadLU.QUADLU_ALPHA_DEFAULT.data.item(),
        elements_max=QuadLU.QUADLU_ALPHA_DEFAULT.data.item(),
    )
)
def test_default_quadlu_forward_equals_x_plus_alpha_squared_else(x: Tensor) -> None:
    assert_close(QuadLU().forward(x), (x + QuadLU.QUADLU_ALPHA_DEFAULT) ** 2)


@given(tensors(elements_min=-0.14, elements_max=0.14), alphas(min_value=0.14))
def test_quadlu_forward_is_correct_else(x: Tensor, alpha: Parameter) -> None:
    assert_close(QuadLU(alpha).forward(x), (x + alpha) ** 2)


@given(tensors())
def test_default_quadlu_forward_is_correct_for_random_input(values: Tensor) -> None:
    less_or_equal_mask = values <= -QuadLU.QUADLU_ALPHA_DEFAULT
    greater_or_equal_mask = values >= QuadLU.QUADLU_ALPHA_DEFAULT
    in_between_mask = ~(less_or_equal_mask | greater_or_equal_mask)
    result_tensor = QuadLU().forward(values)
    assert_equal(result_tensor[less_or_equal_mask].data.numpy(), 0.0)
    assert_close(
        result_tensor[in_between_mask],
        (values[in_between_mask] + QuadLU.QUADLU_ALPHA_DEFAULT) ** 2,
        equal_nan=True,
    )
    assert_close(result_tensor[greater_or_equal_mask], values[greater_or_equal_mask])


@given(tensors(), alphas())
def test_quadlu_forward_is_correct_for_random_input(
    values: Tensor, alpha: Parameter
) -> None:
    less_or_equal_mask = values <= -alpha
    greater_or_equal_mask = values >= alpha
    in_between_mask = ~(less_or_equal_mask | greater_or_equal_mask)
    result_tensor = QuadLU(alpha).forward(values)
    assert_equal(result_tensor[less_or_equal_mask].data.numpy(), 0.0)
    assert_close(
        result_tensor[in_between_mask],
        (values[in_between_mask] + alpha) ** 2,
        equal_nan=True,
    )
    assert_close(
        result_tensor[greater_or_equal_mask],
        4 * alpha * values[greater_or_equal_mask],
        equal_nan=True,
    )
