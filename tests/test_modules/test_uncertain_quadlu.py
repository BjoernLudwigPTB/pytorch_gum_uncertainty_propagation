"""Test the class GUMQuadLU"""
from inspect import signature

import pytest
import torch
from hypothesis import given, HealthCheck, settings
from numpy.testing import assert_equal
from torch import square, Tensor, tensor
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.testing import assert_close  # type: ignore[attr-defined]

from pytorch_gum_uncertainty_propagation import modules
from pytorch_gum_uncertainty_propagation.functionals import (
    QUADLU_ALPHA_DEFAULT,
)
from pytorch_gum_uncertainty_propagation.modules import (
    QuadLU,
    GUMQuadLU,
)
from pytorch_gum_uncertainty_propagation.uncertainties import (
    _is_positive_semi_definite,
    _is_symmetric,
    UncertainTensor,
)
from ..conftest import alphas, tensors, uncertain_tensors


@pytest.fixture
def gum_quadlu_instance() -> GUMQuadLU:
    return GUMQuadLU()


def test_modules_all_contains_uncertain_quadlu() -> None:
    assert GUMQuadLU.__name__ in modules.__all__


def test_modules_actually_contains_uncertain_quadlu() -> None:
    assert hasattr(modules, GUMQuadLU.__name__)


def test_uncertain_quadlu_is_subclass_of_nn_module() -> None:
    assert issubclass(GUMQuadLU, Module)


def test_uncertain_quadlu_has_docstring() -> None:
    assert GUMQuadLU.__doc__ is not None


def test_init_uncertain_quadlu(uncertain_quadlu_instance: GUMQuadLU) -> None:
    assert uncertain_quadlu_instance


def test_uncertain_quadlu_has_parameter_alpha() -> None:
    assert "alpha" in signature(GUMQuadLU).parameters


def test_init_uncertain_quadlu_creates_attribute_alpha(
    uncertain_quadlu_instance: GUMQuadLU,
) -> None:
    assert hasattr(uncertain_quadlu_instance, "_alpha")


def test_init_uncertain_quadlu_creates_attribute_quadlu(
    uncertain_quadlu_instance: GUMQuadLU,
) -> None:
    assert hasattr(uncertain_quadlu_instance, "_quadlu")


def test_uncertain_quadlu_alpha_has_docstring() -> None:
    assert GUMQuadLU._alpha.__doc__ is not None


def test_init_uncertain_quadlu_alpha_requires_grad(
    uncertain_quadlu_instance: GUMQuadLU,
) -> None:
    assert uncertain_quadlu_instance._alpha.requires_grad


def test_init_uncertain_quadlu_contains_constant_for_alphas_default(
    uncertain_quadlu_instance: GUMQuadLU,
) -> None:
    assert hasattr(uncertain_quadlu_instance, "QUADLU_ALPHA_DEFAULT")


def test_init_default_uncertain_quadlu_and_quadlus_creates_same_alpha(
    uncertain_quadlu_instance: GUMQuadLU, quadlu_instance: QuadLU
) -> None:
    assert_close(
        uncertain_quadlu_instance.QUADLU_ALPHA_DEFAULT,
        quadlu_instance.QUADLU_ALPHA_DEFAULT,
    )


def test_init_uncertain_quadlu_creates_alpha_equal_to_default(
    uncertain_quadlu_instance: GUMQuadLU,
) -> None:
    assert_close(
        uncertain_quadlu_instance._alpha, uncertain_quadlu_instance.QUADLU_ALPHA_DEFAULT
    )


@given(alphas())
def test_init_uncertain_quadlu_creates_parameter_alpha(alpha: Parameter) -> None:
    assert hasattr(GUMQuadLU(alpha), "_alpha")


@given(alphas())
def test_init_uncertain_quadlu_with_random_alpha(alpha: Parameter) -> None:
    assert_close(GUMQuadLU(alpha)._alpha, alpha)


def test_uncertain_quadlu_contains_callable_forward() -> None:
    assert callable(GUMQuadLU.forward)


@given(uncertain_tensors(less_than=-GUMQuadLU.QUADLU_ALPHA_DEFAULT.data.item()))
def test_default_uncertain_quadlu_forward_for_small_values(
    uncertain_tensor: UncertainTensor,
) -> None:
    assert_equal(GUMQuadLU().forward(uncertain_tensor)[0].data.numpy(), 0.0)


@given(uncertain_tensors(less_than=-0.1501), alphas(max_value=0.15))
def test_uncertain_quadlu_forward_for_small_values(
    uncertain_tensor: UncertainTensor, alpha: Parameter
) -> None:
    assert_equal(GUMQuadLU(alpha).forward(uncertain_tensor)[0].data.numpy(), 0.0)


@given(tensors())
def test_default_uncertain_quadlu_forward_provides_no_uncertainties_if_not_provided(
    values: Tensor,
) -> None:
    assert GUMQuadLU().forward(UncertainTensor(values, None)).uncertainties is None


@given(uncertain_tensors(greater_than=GUMQuadLU.QUADLU_ALPHA_DEFAULT.data.item()))
def test_default_uncertain_quadlu_forward_for_large_values(
    uncertain_tensor: UncertainTensor,
) -> None:
    assert_close(GUMQuadLU().forward(uncertain_tensor)[0], uncertain_tensor.values)


@given(uncertain_tensors(greater_than=0.15), alphas(max_value=0.15))
def test_uncertain_quadlu_forward_for_large_x(
    uncertain_tensor: UncertainTensor, alpha: Parameter
) -> None:
    assert_close(
        GUMQuadLU(alpha).forward(uncertain_tensor)[0],
        4 * alpha * uncertain_tensor.values,
        equal_nan=True,
    )


@given(
    uncertain_tensors(
        greater_than=-GUMQuadLU.QUADLU_ALPHA_DEFAULT.data.item(),
        less_than=GUMQuadLU.QUADLU_ALPHA_DEFAULT.data.item(),
    )
)
def test_default_uncertain_quadlu_forward_near_zero(
    uncertain_tensor: UncertainTensor,
) -> None:
    assert_close(
        GUMQuadLU().forward(uncertain_tensor)[0],
        square(uncertain_tensor.values + GUMQuadLU.QUADLU_ALPHA_DEFAULT),
    )


@given(uncertain_tensors(greater_than=-0.14, less_than=0.14), alphas(min_value=0.14))
def test_uncertain_quadlu_forward_near_zero(
    uncertain_tensor: UncertainTensor, alpha: Parameter
) -> None:
    assert_close(
        GUMQuadLU(alpha).forward(uncertain_tensor)[0],
        square(uncertain_tensor.values + alpha),
    )


@given(uncertain_tensors())
def test_default_uncertain_quadlu_forward_values_for_random_input(
    uncertain_tensor: UncertainTensor,
) -> None:
    values = uncertain_tensor.values
    less_or_equal_mask = values <= -GUMQuadLU.QUADLU_ALPHA_DEFAULT
    greater_or_equal_mask = values >= GUMQuadLU.QUADLU_ALPHA_DEFAULT
    in_between_mask = ~(less_or_equal_mask | greater_or_equal_mask)
    result_tensor = GUMQuadLU().forward(uncertain_tensor)[0]
    assert_equal(result_tensor[less_or_equal_mask].data.numpy(), 0.0)
    assert_close(
        result_tensor[in_between_mask],
        square(values[in_between_mask] + GUMQuadLU.QUADLU_ALPHA_DEFAULT),
        equal_nan=True,
    )
    assert_close(result_tensor[greater_or_equal_mask], values[greater_or_equal_mask])


@given(uncertain_tensors(), alphas())
def test_uncertain_quadlu_forward_values_for_random_input(
    uncertain_tensor: UncertainTensor, alpha: Parameter
) -> None:
    values = uncertain_tensor.values
    less_or_equal_mask = values <= -alpha
    greater_or_equal_mask = values >= alpha
    in_between_mask = ~(less_or_equal_mask | greater_or_equal_mask)
    result_tensor = GUMQuadLU(alpha).forward(uncertain_tensor)[0]
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


@given(uncertain_tensors(greater_than=QUADLU_ALPHA_DEFAULT.data.item()))
def test_default_uncertain_quadlu_forward_accepts_big_input(
    values_and_uncertainties: UncertainTensor,
) -> None:
    assert GUMQuadLU().forward(values_and_uncertainties)


@given(uncertain_tensors(greater_than=0.16), alphas(max_value=0.16))
def test_uncertain_quadlu_forward_accepts_big_input(
    values_and_uncertainties: UncertainTensor, alpha: Parameter
) -> None:
    assert GUMQuadLU(alpha).forward(values_and_uncertainties)


@given(
    uncertain_tensors(
        greater_than=-QUADLU_ALPHA_DEFAULT.data.item(),
        less_than=QUADLU_ALPHA_DEFAULT.data.item(),
    )
)
def test_default_uncertain_quadlu_forward_accepts_medium_input(
    values_and_uncertainties: UncertainTensor,
) -> None:
    assert GUMQuadLU().forward(values_and_uncertainties)


@given(
    uncertain_tensors(greater_than=-0.14, less_than=0.14),
    alphas(max_value=0.14),
)
def test_uncertain_quadlu_forward_accepts_medium_input(
    values_and_uncertainties: UncertainTensor, alpha: Parameter
) -> None:
    assert GUMQuadLU(alpha).forward(values_and_uncertainties)


@given(uncertain_tensors(less_than=-QUADLU_ALPHA_DEFAULT.data.item()))
def test_default_uncertain_quadlu_forward_accepts_small_input(
    values_and_uncertainties: UncertainTensor,
) -> None:
    assert GUMQuadLU().forward(values_and_uncertainties)


@given(uncertain_tensors(less_than=-0.14), alphas(max_value=0.14))
def test_uncertain_quadlu_forward_accepts_small_input(
    values_and_uncertainties: UncertainTensor, alpha: Parameter
) -> None:
    assert GUMQuadLU(alpha).forward(values_and_uncertainties)


@given(uncertain_tensors())
def test_default_uncertain_quadlu_forward_accepts_random_input(
    values_and_uncertainties: UncertainTensor,
) -> None:
    assert GUMQuadLU().forward(values_and_uncertainties)


@given(uncertain_tensors(less_than=-QUADLU_ALPHA_DEFAULT.data.item()))
def test_default_uncertain_quadlu_forward_uncertainties_for_small_input(
    values_and_uncertainties: UncertainTensor,
) -> None:
    assert values_and_uncertainties.uncertainties is not None
    assert_close(
        GUMQuadLU().forward(values_and_uncertainties).uncertainties,
        values_and_uncertainties.uncertainties.zero_(),
    )


@given(uncertain_tensors(less_than=-0.13), alphas(max_value=0.13))
def test_uncertain_quadlu_forward_uncertainties_for_small_input(
    values_and_uncertainties: UncertainTensor, alpha: Parameter
) -> None:
    assert values_and_uncertainties.uncertainties is not None
    assert_close(
        GUMQuadLU(alpha).forward(values_and_uncertainties).uncertainties,
        values_and_uncertainties.uncertainties.zero_(),
    )


@given(uncertain_tensors(greater_than=QUADLU_ALPHA_DEFAULT.data.item()))
def test_default_uncertain_quadlu_forward_uncertainties_for_large_input(
    values_and_uncertainties: UncertainTensor,
) -> None:
    assert_close(
        GUMQuadLU().forward(values_and_uncertainties).uncertainties,
        values_and_uncertainties.uncertainties,
    )


@given(uncertain_tensors(greater_than=0.1401), alphas(max_value=0.14))
def test_uncertain_quadlu_forward_uncertainties_for_large_input(
    values_and_uncertainties: UncertainTensor, alpha: Parameter
) -> None:
    assert values_and_uncertainties.uncertainties is not None
    result_uncertainties = (
        GUMQuadLU(alpha)
        .forward(
            values_and_uncertainties,
        )
        .uncertainties
    )
    assert result_uncertainties is not None
    assert_close(
        GUMQuadLU(alpha)
        .forward(
            values_and_uncertainties,
        )
        .uncertainties,
        values_and_uncertainties.uncertainties * 16 * square(alpha),
    )


@given(
    uncertain_tensors(
        greater_than=-QUADLU_ALPHA_DEFAULT.data.item(),
        less_than=QUADLU_ALPHA_DEFAULT.data.item(),
    )
)
def test_default_uncertain_quadlu_forward_uncertainties_for_medium_input(
    values_and_uncertainties: UncertainTensor,
) -> None:
    first_derivs = 2 * values_and_uncertainties.values + 0.5
    assert_close(
        GUMQuadLU()
        .forward(
            values_and_uncertainties,
        )
        .uncertainties,
        first_derivs
        * values_and_uncertainties.uncertainties
        * first_derivs.unsqueeze(1),
    )


@given(
    uncertain_tensors(greater_than=-0.14, less_than=0.14),
    alphas(min_value=0.14),
)
def test_uncertain_quadlu_forward_uncertainties_for_medium_input(
    values_and_uncertainties: UncertainTensor, alpha: Parameter
) -> None:
    first_derivs = 2 * (values_and_uncertainties.values + alpha)
    assert_close(
        GUMQuadLU(alpha)
        .forward(
            values_and_uncertainties,
        )
        .uncertainties,
        first_derivs
        * values_and_uncertainties.uncertainties
        * first_derivs.unsqueeze(1),
    )


def test_default_uncertain_quadlu_forward_uncertainties_for_given_input() -> None:
    uncertain_tensor = UncertainTensor(tensor([-1.0, 0.0, 1.0]), torch.eye(3))
    assert_close(
        GUMQuadLU().forward(uncertain_tensor).uncertainties,
        tensor([[0.0, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 1.0]]),
    )


@given(alphas(max_value=0.12))
def test_uncertain_quadlu_forward_uncertainties_for_given_input(
    alpha: Parameter,
) -> None:
    uncertain_tensor = UncertainTensor(
        tensor([1.0, 0.0, -1.0]),
        torch.tensor(
            [
                [0.0625, 0.0, 0.0],
                [0.0, 0.25, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    )
    assert_close(
        GUMQuadLU(alpha).forward(uncertain_tensor).uncertainties,
        tensor([[square(alpha), 0.0, 0.0], [0.0, square(alpha), 0.0], [0.0, 0.0, 0.0]]),
    )


@given(uncertain_tensors())
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_gum_quadlu_forward_results_in_positive_semi_definite_uncertainties(
    gum_quadlu_instance: GUMQuadLU,
    uncertain_tensor: UncertainTensor,
) -> None:
    result_uncertainties = gum_quadlu_instance.forward(uncertain_tensor).uncertainties
    assert result_uncertainties is not None
    assert _is_positive_semi_definite(result_uncertainties)


@given(uncertain_tensors())
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_gum_quadlu_forward_results_in_symmetric_uncertainties(
    gum_quadlu_instance: GUMQuadLU,
    uncertain_tensor: UncertainTensor,
) -> None:
    result_uncertainties = gum_quadlu_instance.forward(uncertain_tensor).uncertainties
    assert result_uncertainties is not None
    assert _is_symmetric(result_uncertainties)
