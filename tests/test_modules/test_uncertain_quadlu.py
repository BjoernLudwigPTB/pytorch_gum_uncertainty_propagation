"""Test the class UncertainQuadLU"""
from hypothesis import given, strategies as hst
from numpy.testing import assert_equal
from torch import Tensor
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.testing import assert_close  # type: ignore[attr-defined]

from gum_compliant_neural_network_uncertainty_propagation import modules
from gum_compliant_neural_network_uncertainty_propagation.modules import (
    QuadLU,
    UncertainQuadLU,
)
from ..conftest import alphas, tensors


def test_modules_all_contains_quadlu() -> None:
    assert UncertainQuadLU.__name__ in modules.__all__


def test_init_uncertain_quadlu(uncertain_quadlu: UncertainQuadLU) -> None:
    assert uncertain_quadlu


def test_init_uncertain_quadlu_creates_parameter(
    uncertain_quadlu: UncertainQuadLU,
) -> None:
    assert hasattr(uncertain_quadlu, "_alpha")


def test_init_uncertain_quadlu_alpha_requires_grad(
    uncertain_quadlu: UncertainQuadLU,
) -> None:
    assert uncertain_quadlu._alpha.requires_grad


def test_init_uncertain_quadlu_contains_constant_for_alphas_default(
    uncertain_quadlu: UncertainQuadLU,
) -> None:
    assert hasattr(uncertain_quadlu, "QUADLU_ALPHA_DEFAULT")


def test_init_uncertain_quadlu_constant_for_alphas_default_equals_one(
    uncertain_quadlu: UncertainQuadLU, quadlu_instance: QuadLU
) -> None:
    assert_close(
        uncertain_quadlu.QUADLU_ALPHA_DEFAULT, quadlu_instance.QUADLU_ALPHA_DEFAULT
    )


def test_init_uncertain_quadlu_creates_alpha_equal_to_default(
    uncertain_quadlu: UncertainQuadLU,
) -> None:
    assert_close(uncertain_quadlu._alpha, uncertain_quadlu.QUADLU_ALPHA_DEFAULT)


def test_init_uncertain_quadlu_creates_parameter_with_sensible_default() -> None:
    assert hasattr(UncertainQuadLU(), "_alpha")


def test_uncertain_quadlu_is_subclass_of_nn_module() -> None:
    assert issubclass(UncertainQuadLU, Module)


def test_uncertain_quadlu_has_docstring() -> None:
    assert UncertainQuadLU.__doc__ is not None


@given(alphas())
def test_init_uncertain_quadlu_with_anonymous_alpha(alpha: Parameter) -> None:
    assert_close(UncertainQuadLU(alpha)._alpha, alpha)


@given(tensors())
def test_uncertain_quadlu_contains_custom_forward(x: Tensor) -> None:
    assert UncertainQuadLU().forward(x)


@given(tensors(), hst.integers(min_value=1, max_value=10))
def test_uncertain_quadlu_apply_repeated_custom_forward(x: Tensor, iters: int) -> None:
    tmp_uncertain_quadlu = UncertainQuadLU()
    for _ in range(iters):
        assert tmp_uncertain_quadlu.forward(*tmp_uncertain_quadlu.forward(x, x))


@given(tensors(elements_max=-UncertainQuadLU.QUADLU_ALPHA_DEFAULT.data.item()))
def test_init_uncertain_quadlu_forward_equals_zero_for_small_x(x: Tensor) -> None:
    assert_equal(UncertainQuadLU().forward(x)[0].data.numpy(), 0.0)


@given(tensors())
def test_uncertain_quadlu_forward_provides_no_uncertainties_if_not_provided(
    x: Tensor,
) -> None:
    assert UncertainQuadLU().forward(x)[1] is None


@given(
    tensors(
        elements_min=UncertainQuadLU.QUADLU_ALPHA_DEFAULT.data.item(), elements_max=1e30
    )
)
def test_uncertain_quadlu_with_default_alpha_forward_equals_x_for_large_x(
    x: Tensor,
) -> None:
    assert_close(UncertainQuadLU().forward(x)[0], x)


@given(
    tensors(
        elements_min=-UncertainQuadLU.QUADLU_ALPHA_DEFAULT.data.item(),
        elements_max=UncertainQuadLU.QUADLU_ALPHA_DEFAULT.data.item(),
    )
)
def test_uncertain_quadlu_with_default_alpha_forward_equals_x_plus_alpha_squared_else(
    x: Tensor,
) -> None:
    assert_close(
        UncertainQuadLU().forward(x)[0],
        (x + UncertainQuadLU.QUADLU_ALPHA_DEFAULT) ** 2,
        rtol=2e-7,
        atol=4e-7,
    )


@given(
    tensors(
        elements_min=-UncertainQuadLU.QUADLU_ALPHA_DEFAULT.data.item(),
        elements_max=UncertainQuadLU.QUADLU_ALPHA_DEFAULT.data.item(),
    )
)
def test_uncertain_quadlu_forward_equals_x_plus_alpha_squared_else(x: Tensor) -> None:
    assert_close(
        UncertainQuadLU().forward(x)[0], (x + UncertainQuadLU.QUADLU_ALPHA_DEFAULT) ** 2
    )
