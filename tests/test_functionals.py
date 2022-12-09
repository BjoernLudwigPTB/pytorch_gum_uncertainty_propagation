import torch
from hypothesis import given, strategies as hst
from numpy.testing import assert_allclose, assert_equal
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
def test_quadlu_with_default_alpha_is_tensor_for_0_dim(values: float) -> None:
    assert isinstance(quadlu(tensor(values)), Tensor)


@given(hst.floats(), alphas())
def test_quadlu_is_tensor_for_0_dims(values: float, alpha: Parameter) -> None:
    assert isinstance(quadlu(tensor(values), alpha), Tensor)


@given(tensors())
def test_quadlu_with_default_alpha_is_tensor_for_tensor(
    values: Tensor,
) -> None:
    assert isinstance(quadlu(values), Tensor)


@given(tensors(), alphas())
def test_quadlu_is_tensor_for_tensor(values: Tensor, alpha: Parameter) -> None:
    assert isinstance(quadlu(values, alpha), Tensor)


@given(tensors())
def test_quadlu_with_default_alpha_keeps_shape(values: Tensor) -> None:
    assert_equal(quadlu(values).shape, values.shape)


@given(tensors(), alphas())
def test_quadlu_keeps_shape(values: Tensor, alpha: Parameter) -> None:
    assert_equal(quadlu(values, alpha).shape, values.shape)


@given(tensors(elements_max=-QUADLU_ALPHA_DEFAULT.data.item()))
def test_quadlu_with_default_alpha_is_zero_for_small_values(values: Tensor) -> None:
    assert_allclose(quadlu(values).data.numpy(), 0.0)


@given(tensors(elements_max=-0.301), alphas(max_value=0.3))
def test_quadlu_is_zero_for_small_values(values: Tensor, alpha: Parameter) -> None:
    assert_allclose(quadlu(values, alpha).data.numpy(), 0.0)


@given(tensors(elements_min=QUADLU_ALPHA_DEFAULT.data.item()))
def test_quadlu_with_default_alpha_is_linear_for_big_values(values: Tensor) -> None:
    assert_allclose(quadlu(values).data.numpy(), values)


@given(
    tensors(elements_min=0.401, elements_max=10),
    alphas(min_value=1e-2, max_value=0.4),
)
def test_quadlu_is_linear_for_big_values(values: Tensor, alpha: Parameter) -> None:
    assert_allclose(
        quadlu(values, alpha).data.numpy(), 4 * alpha.detach().numpy() * values
    )


@given(
    tensors(
        elements_min=-QUADLU_ALPHA_DEFAULT.data.item(),
        elements_max=QUADLU_ALPHA_DEFAULT.data.item(),
    )
)
def test_quadlu_with_default_alpha_is_quadratic_for_medium(values: Tensor) -> None:
    assert_allclose(
        quadlu(values).data.numpy(),
        (values + QUADLU_ALPHA_DEFAULT.data.item()) ** 2,
    )


@given(
    tensors(elements_min=-0.5, elements_max=0.5),
    alphas(min_value=0.501, exclude_min=False),
)
def test_quadlu_is_quadratic_for_medium(values: Tensor, alpha: Parameter) -> None:
    assert_allclose(
        quadlu(values, alpha).data.numpy(),
        (values + alpha.data.item()) ** 2,
        rtol=4e-5,
    )


@given(
    tensors(elements_max=-QUADLU_ALPHA_DEFAULT.data.item()),
    tensors(
        elements_min=-QUADLU_ALPHA_DEFAULT.data.item(),
        elements_max=QUADLU_ALPHA_DEFAULT.data.item(),
    ),
    tensors(elements_min=QUADLU_ALPHA_DEFAULT.data.item()),
)
def test_quadlu_with_default_alpha_is_correct(
    small_values: Tensor,
    medium_values: Tensor,
    big_values: Tensor,
) -> None:
    input_tensor = torch.concat((small_values, medium_values, big_values))
    assert_allclose(quadlu(input_tensor)[: len(small_values)].data.numpy(), 0.0)
    assert_close(
        quadlu(input_tensor)[
            len(small_values) : len(small_values) + len(medium_values)
        ],
        (medium_values + QUADLU_ALPHA_DEFAULT) ** 2,
    )
    assert_close(quadlu(input_tensor)[-len(big_values) :], big_values)
