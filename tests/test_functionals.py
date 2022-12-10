import torch
from hypothesis import given, strategies as hst
from numpy.testing import assert_equal
from torch import tensor, Tensor
from torch.nn.parameter import Parameter
from torch.testing import assert_close  # type: ignore[attr-defined]

from gum_compliant_neural_network_uncertainty_propagation import functionals
from gum_compliant_neural_network_uncertainty_propagation.functionals import (
    quadlu,
    QUADLU_ALPHA_DEFAULT,
)
from .conftest import alphas, tensors


def test_functionals_has_default_alpha() -> None:
    assert hasattr(functionals, "QUADLU_ALPHA_DEFAULT")


def test_functionals_has_module_docstring() -> None:
    assert functionals.__doc__ is not None


def test_functionals_all_contains_quadlu() -> None:
    assert quadlu.__name__ in functionals.__all__


@given(hst.floats())
def test_default_quadlu_is_tensor_for_0_dim(values: float) -> None:
    assert isinstance(quadlu(tensor(values)), Tensor)


@given(hst.floats(), alphas())
def test_quadlu_is_tensor_for_0_dims(values: float, alpha: Parameter) -> None:
    assert isinstance(quadlu(tensor(values), alpha), Tensor)


@given(tensors())
def test_default_quadlu_is_tensor_for_tensor(values: Tensor) -> None:
    assert isinstance(quadlu(values), Tensor)


@given(tensors(), alphas())
def test_quadlu_is_tensor_for_tensor(values: Tensor, alpha: Parameter) -> None:
    assert isinstance(quadlu(values, alpha), Tensor)


@given(tensors())
def test_default_quadlu_keeps_shape(values: Tensor) -> None:
    assert_equal(quadlu(values).shape, values.shape)


@given(tensors(), alphas())
def test_quadlu_keeps_shape(values: Tensor, alpha: Parameter) -> None:
    assert_equal(quadlu(values, alpha).shape, values.shape)


@given(tensors(elements_max=-QUADLU_ALPHA_DEFAULT.data.item()))
def test_default_quadlu_forward_is_correct_for_small_values(values: Tensor) -> None:
    assert_equal(quadlu(values).data.numpy(), 0.0)


@given(tensors(elements_max=-0.201), alphas(max_value=0.2))
def test_quadlu_forward_is_correct_for_small_values(
    values: Tensor, alpha: Parameter
) -> None:
    assert_equal(quadlu(values, alpha).data.numpy(), 0.0)


@given(tensors(elements_min=QUADLU_ALPHA_DEFAULT.data.item()))
def test_default_quadlu_forward_is_correct_for_bigs(values: Tensor) -> None:
    assert_equal(quadlu(values).data.numpy(), values)


@given(
    tensors(elements_min=0.401, elements_max=10), alphas(min_value=1e-2, max_value=0.4)
)
def test_quadlu_forward_is_correct_for_bigs(values: Tensor, alpha: Parameter) -> None:
    assert_equal(
        quadlu(values, alpha).data.numpy(), 4 * alpha.detach().numpy() * values
    )


@given(
    tensors(
        elements_min=-QUADLU_ALPHA_DEFAULT.data.item(),
        elements_max=QUADLU_ALPHA_DEFAULT.data.item(),
    )
)
def test_default_quadlu_forward_is_correct_for_mediums(values: Tensor) -> None:
    assert_equal(
        quadlu(values).data.numpy(),
        torch.square(values + QUADLU_ALPHA_DEFAULT.data.item()),
    )


@given(tensors(elements_min=-0.5, elements_max=0.5), alphas(min_value=0.5))
def test_quadlu_forward_is_correct_for_medium(values: Tensor, alpha: Parameter) -> None:
    assert_equal(quadlu(values, alpha).data.numpy(), (values + alpha.data.item()) ** 2)


@given(
    tensors(elements_max=-QUADLU_ALPHA_DEFAULT.data.item()),
    tensors(
        elements_min=-QUADLU_ALPHA_DEFAULT.data.item(),
        elements_max=QUADLU_ALPHA_DEFAULT.data.item(),
    ),
    tensors(elements_min=QUADLU_ALPHA_DEFAULT.data.item()),
)
def test_default_quadlu_forward_is_correct_for_sorted_input(
    smalls: Tensor, mediums: Tensor, bigs: Tensor
) -> None:
    input_tensor = torch.concat((smalls, mediums, bigs))
    assert_equal(quadlu(input_tensor)[: len(smalls)].data.numpy(), 0.0)
    assert_close(
        quadlu(input_tensor)[len(smalls) : len(smalls) + len(mediums)],
        torch.square(mediums + QUADLU_ALPHA_DEFAULT),
    )
    assert_close(quadlu(input_tensor)[-len(bigs) :], bigs)


@given(tensors())
def test_default_quadlu_forward_is_correct_for_random_input(values: Tensor) -> None:
    less_or_equal_mask = values <= -QUADLU_ALPHA_DEFAULT
    greater_or_equal_mask = values >= QUADLU_ALPHA_DEFAULT
    in_between_mask = ~(less_or_equal_mask | greater_or_equal_mask)
    result_tensor = quadlu(values)
    assert_equal(result_tensor[less_or_equal_mask].data.numpy(), 0.0)
    assert_close(
        result_tensor[in_between_mask],
        torch.square(values[in_between_mask] + QUADLU_ALPHA_DEFAULT),
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
    result_tensor = quadlu(values, alpha)
    assert_equal(result_tensor[less_or_equal_mask].data.numpy(), 0.0)
    assert_close(
        result_tensor[in_between_mask],
        torch.square(values[in_between_mask] + alpha),
        equal_nan=True,
    )
    assert_close(
        result_tensor[greater_or_equal_mask],
        4 * alpha * values[greater_or_equal_mask],
        equal_nan=True,
    )
