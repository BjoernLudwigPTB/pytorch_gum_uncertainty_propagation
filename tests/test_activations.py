import pytest
from hypothesis import given, strategies as hst
from numpy.testing import assert_almost_equal, assert_equal
from torch.nn import Module

from GUM_compliant_neural_network_uncertainty_propagation import activations
from GUM_compliant_neural_network_uncertainty_propagation.activations import QuadLU


@pytest.fixture()
def quadlu() -> QuadLU:
    return QuadLU()


def test_activations_has_module_docstring():
    assert activations.__doc__ is not None


def test_init_quadlu(quadlu: QuadLU):
    assert quadlu


def test_init_quadlu_creates_parameter(quadlu: QuadLU):
    assert hasattr(quadlu, "_alpha")


def test_init_quadlu_alpha_requires_grad(quadlu: QuadLU):
    assert quadlu._alpha.requires_grad


def test_init_quadlu_contains_constant_for_alphas_default(quadlu: QuadLU):
    assert hasattr(quadlu, "ALPHA_DEFAULT")


def test_init_quadlu_constant_for_alphas_default_equals_one(quadlu: QuadLU):
    assert_equal(quadlu.ALPHA_DEFAULT, 1.0)


def test_init_quadlu_creates_alpha_equal_to_default(quadlu: QuadLU):
    assert_equal(quadlu._alpha.data.item(), quadlu.ALPHA_DEFAULT)


def test_init_quadlu_creates_parameter_with_sensible_default():
    assert hasattr(QuadLU(), "_alpha")


def test_quadlu_is_subclass_of_nn_module():
    assert issubclass(QuadLU, Module)


def test_quadlu_has_docstring():
    assert QuadLU.__doc__ is not None


@given(
    hst.floats(
        min_value=0,
        max_value=1,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
        exclude_min=True,
    )
)
def test_init_quadlu_with_anonymous_alpha(alpha):
    assert_almost_equal(QuadLU(alpha)._alpha.data.item(), alpha)
