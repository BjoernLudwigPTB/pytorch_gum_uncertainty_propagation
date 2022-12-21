"""Test the class UncertainQuadLU"""
from inspect import signature

import torch
from hypothesis import given
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
    UncertainQuadLU,
)
from pytorch_gum_uncertainty_propagation.uncertainties import (
    UncertainTensor,
)
from ..conftest import alphas, tensors, uncertain_tensors


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


def test_init_uncertain_quadlu_creates_attribute_quadlu(
    uncertain_quadlu_instance: UncertainQuadLU,
) -> None:
    assert hasattr(uncertain_quadlu_instance, "_quadlu")


def test_uncertain_quadlu_alpha_has_docstring() -> None:
    assert UncertainQuadLU._alpha.__doc__ is not None


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


@given(uncertain_tensors(less_than=-UncertainQuadLU.QUADLU_ALPHA_DEFAULT.data.item()))
def test_default_uncertain_quadlu_forward_for_small_values(
    uncertain_tensor: UncertainTensor,
) -> None:
    assert_equal(UncertainQuadLU().forward(uncertain_tensor)[0].data.numpy(), 0.0)


@given(uncertain_tensors(less_than=-0.1501), alphas(max_value=0.15))
def test_uncertain_quadlu_forward_for_small_values(
    uncertain_tensor: UncertainTensor, alpha: Parameter
) -> None:
    assert_equal(UncertainQuadLU(alpha).forward(uncertain_tensor)[0].data.numpy(), 0.0)


@given(tensors())
def test_default_uncertain_quadlu_forward_provides_no_uncertainties_if_not_provided(
    values: Tensor,
) -> None:
    assert (
        UncertainQuadLU().forward(UncertainTensor(values, None)).uncertainties is None
    )


@given(uncertain_tensors(greater_than=UncertainQuadLU.QUADLU_ALPHA_DEFAULT.data.item()))
def test_default_uncertain_quadlu_forward_for_large_values(
    uncertain_tensor: UncertainTensor,
) -> None:
    assert_close(
        UncertainQuadLU().forward(uncertain_tensor)[0], uncertain_tensor.values
    )


@given(uncertain_tensors(greater_than=0.15), alphas(max_value=0.15))
def test_uncertain_quadlu_forward_for_large_x(
    uncertain_tensor: UncertainTensor, alpha: Parameter
) -> None:
    assert_close(
        UncertainQuadLU(alpha).forward(uncertain_tensor)[0],
        4 * alpha * uncertain_tensor.values,
        equal_nan=True,
    )


@given(
    uncertain_tensors(
        greater_than=-UncertainQuadLU.QUADLU_ALPHA_DEFAULT.data.item(),
        less_than=UncertainQuadLU.QUADLU_ALPHA_DEFAULT.data.item(),
    )
)
def test_default_uncertain_quadlu_forward_near_zero(
    uncertain_tensor: UncertainTensor,
) -> None:
    assert_close(
        UncertainQuadLU().forward(uncertain_tensor)[0],
        square(uncertain_tensor.values + UncertainQuadLU.QUADLU_ALPHA_DEFAULT),
    )


@given(uncertain_tensors(greater_than=-0.14, less_than=0.14), alphas(min_value=0.14))
def test_uncertain_quadlu_forward_near_zero(
    uncertain_tensor: UncertainTensor, alpha: Parameter
) -> None:
    assert_close(
        UncertainQuadLU(alpha).forward(uncertain_tensor)[0],
        square(uncertain_tensor.values + alpha),
    )


@given(uncertain_tensors())
def test_default_uncertain_quadlu_forward_values_for_random_input(
    uncertain_tensor: UncertainTensor,
) -> None:
    values = uncertain_tensor.values
    less_or_equal_mask = values <= -UncertainQuadLU.QUADLU_ALPHA_DEFAULT
    greater_or_equal_mask = values >= UncertainQuadLU.QUADLU_ALPHA_DEFAULT
    in_between_mask = ~(less_or_equal_mask | greater_or_equal_mask)
    result_tensor = UncertainQuadLU().forward(uncertain_tensor)[0]
    assert_equal(result_tensor[less_or_equal_mask].data.numpy(), 0.0)
    assert_close(
        result_tensor[in_between_mask],
        square(values[in_between_mask] + UncertainQuadLU.QUADLU_ALPHA_DEFAULT),
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
    result_tensor = UncertainQuadLU(alpha).forward(uncertain_tensor)[0]
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
    assert UncertainQuadLU().forward(values_and_uncertainties)


@given(uncertain_tensors(greater_than=0.16), alphas(max_value=0.16))
def test_uncertain_quadlu_forward_accepts_big_input(
    values_and_uncertainties: UncertainTensor, alpha: Parameter
) -> None:
    assert UncertainQuadLU(alpha).forward(values_and_uncertainties)


@given(
    uncertain_tensors(
        greater_than=-QUADLU_ALPHA_DEFAULT.data.item(),
        less_than=QUADLU_ALPHA_DEFAULT.data.item(),
    )
)
def test_default_uncertain_quadlu_forward_accepts_medium_input(
    values_and_uncertainties: UncertainTensor,
) -> None:
    assert UncertainQuadLU().forward(values_and_uncertainties)


@given(
    uncertain_tensors(greater_than=-0.14, less_than=0.14),
    alphas(max_value=0.14),
)
def test_uncertain_quadlu_forward_accepts_medium_input(
    values_and_uncertainties: UncertainTensor, alpha: Parameter
) -> None:
    assert UncertainQuadLU(alpha).forward(values_and_uncertainties)


@given(uncertain_tensors(less_than=-QUADLU_ALPHA_DEFAULT.data.item()))
def test_default_uncertain_quadlu_forward_accepts_small_input(
    values_and_uncertainties: UncertainTensor,
) -> None:
    assert UncertainQuadLU().forward(values_and_uncertainties)


@given(uncertain_tensors(less_than=-0.14), alphas(max_value=0.14))
def test_uncertain_quadlu_forward_accepts_small_input(
    values_and_uncertainties: UncertainTensor, alpha: Parameter
) -> None:
    assert UncertainQuadLU(alpha).forward(values_and_uncertainties)


@given(uncertain_tensors())
def test_default_uncertain_quadlu_forward_accepts_random_input(
    values_and_uncertainties: UncertainTensor,
) -> None:
    assert UncertainQuadLU().forward(values_and_uncertainties)


@given(uncertain_tensors(less_than=-QUADLU_ALPHA_DEFAULT.data.item()))
def test_default_uncertain_quadlu_forward_uncertainties_for_small_input(
    values_and_uncertainties: UncertainTensor,
) -> None:
    assert values_and_uncertainties.uncertainties is not None
    assert_close(
        UncertainQuadLU().forward(values_and_uncertainties).uncertainties,
        values_and_uncertainties.uncertainties.zero_(),
    )


@given(uncertain_tensors(less_than=-0.13), alphas(max_value=0.13))
def test_uncertain_quadlu_forward_uncertainties_for_small_input(
    values_and_uncertainties: UncertainTensor, alpha: Parameter
) -> None:
    assert values_and_uncertainties.uncertainties is not None
    assert_close(
        UncertainQuadLU(alpha).forward(values_and_uncertainties).uncertainties,
        values_and_uncertainties.uncertainties.zero_(),
    )


@given(uncertain_tensors(greater_than=QUADLU_ALPHA_DEFAULT.data.item()))
def test_default_uncertain_quadlu_forward_uncertainties_for_large_input(
    values_and_uncertainties: UncertainTensor,
) -> None:
    assert_close(
        UncertainQuadLU().forward(values_and_uncertainties).uncertainties,
        values_and_uncertainties.uncertainties,
    )


@given(uncertain_tensors(greater_than=0.1401), alphas(max_value=0.14))
def test_uncertain_quadlu_forward_uncertainties_for_large_input(
    values_and_uncertainties: UncertainTensor, alpha: Parameter
) -> None:
    assert values_and_uncertainties.uncertainties is not None
    result_uncertainties = (
        UncertainQuadLU(alpha)
        .forward(
            values_and_uncertainties,
        )
        .uncertainties
    )
    assert result_uncertainties is not None
    assert_close(
        UncertainQuadLU(alpha)
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
        UncertainQuadLU()
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
        UncertainQuadLU(alpha)
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
        UncertainQuadLU().forward(uncertain_tensor).uncertainties,
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
        UncertainQuadLU(alpha).forward(uncertain_tensor).uncertainties,
        tensor([[square(alpha), 0.0, 0.0], [0.0, square(alpha), 0.0], [0.0, 0.0, 0.0]]),
    )
