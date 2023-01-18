from inspect import signature
from typing import cast

import numpy as np
import torch
from hypothesis import assume, given, strategies as hst
from hypothesis.extra import numpy as hnp
from hypothesis.strategies import composite, DrawFn, SearchStrategy
from numpy import array
from numpy.testing import assert_equal
from torch import diag, isnan, tensor, Tensor
from torch.testing import assert_close  # type: ignore[attr-defined]

from pytorch_gum_uncertainty_propagation import uncertainties
from pytorch_gum_uncertainty_propagation.uncertainties import (
    _is_positive_semi_definite,
    _is_symmetric,
    cov_matrix_from_std_uncertainties,
    UncertainTensor,
)
from .conftest import tensors, uncertain_tensors


@composite
def square_tensors(
    draw: DrawFn, dimen: int | None = None, symmetric: bool = False
) -> SearchStrategy[Tensor]:
    rows: int = (
        draw(hst.integers(min_value=2, max_value=10)) if dimen is None else dimen
    )
    if symmetric:
        result_tensor = diag(tensor(draw(hnp.arrays(float, rows))))
        for i_diag in range(1, rows):
            diagonal_values = tensor(draw(hnp.arrays(float, rows - i_diag)))
            result_tensor += torch.diag(diagonal_values, i_diag)
            result_tensor += torch.diag(diagonal_values, -i_diag)
    else:
        result_tensor = tensor(draw(hnp.arrays(float, (rows, rows), unique=True)))
        non_symmetric = False
        for i_row in range(rows):
            for i_column in range(i_row):
                if not (
                    isnan(result_tensor[i_row, i_column])
                    and isnan(result_tensor[i_column, i_row])
                ) and not torch.isclose(
                    result_tensor[i_row, i_column],
                    result_tensor[i_column, i_row],
                    rtol=1e-5,
                ):
                    non_symmetric = True
                    break
            if non_symmetric:
                break
        assume(non_symmetric)
    return cast(SearchStrategy[Tensor], result_tensor)


@composite
def sigmas(draw: DrawFn) -> SearchStrategy[Tensor]:
    array_strategy = hnp.arrays(
        float,
        shape=hnp.array_shapes(min_dims=1, max_dims=1, min_side=1, max_side=100),
        elements=hst.floats(
            allow_subnormal=False,
            allow_nan=False,
            allow_infinity=False,
            min_value=0.1,
            max_value=10,
        ),
    )
    sigma = tensor(draw(array_strategy))
    return cast(SearchStrategy[Tensor], sigma)


def test_modules_has_attribute_uncertain_tensor() -> None:
    assert hasattr(uncertainties, "UncertainTensor")


def test_uncertain_tensor_present_in_all() -> None:
    assert UncertainTensor.__name__ in uncertainties.__all__


def test_uncertainties_has_attribute_uncertain_tensor() -> None:
    assert hasattr(uncertainties, "cov_matrix_from_std_uncertainties")


def test_cov_matrix_from_std_uncertainties_present_in_all() -> None:
    assert cov_matrix_from_std_uncertainties.__name__ in uncertainties.__all__


def test_uncertainties_has_attribute_is_positive_semi_definite() -> None:
    assert hasattr(uncertainties, "_is_positive_semi_definite")


def test_uncertainties_has_attribute_is_symmetric() -> None:
    assert hasattr(uncertainties, "_is_symmetric")


@given(uncertain_tensors())
def test_is_positive_semi_definite_usual_call(
    uncertain_tensor: UncertainTensor,
) -> None:
    assert uncertain_tensor.uncertainties is not None
    assert isinstance(
        _is_positive_semi_definite(uncertain_tensor.uncertainties).item(), bool
    )


@given(hst.integers(min_value=1, max_value=10))
def test_is_positive_semi_definite_against_zero(length: int) -> None:
    assert _is_positive_semi_definite(tensor(np.zeros((length, length))))


def test_is_positive_semi_definite_for_single_instance_true() -> None:
    A = tensor([[5.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 3.0]])
    assert _is_positive_semi_definite(A) == tensor(True)


def test_is_positive_semi_definite_for_single_instance_false() -> None:
    A = tensor(array([[1.0, 2.0, 1.0], [2.0, 2.0, 2.0], [1.0, 2.0, 3.0]]))
    assert _is_positive_semi_definite(A) == tensor(False)


@given(sigmas())
def test_cov_matrix_from_std_uncertainties_returns_tensor(sigma: Tensor) -> None:
    assert isinstance(cov_matrix_from_std_uncertainties(sigma), Tensor)


@given(sigmas())
def test_cov_matrix_from_std_uncertainties_returns_correct_shape(sigma: Tensor) -> None:
    assert_equal(
        cov_matrix_from_std_uncertainties(sigma).shape, (len(sigma), len(sigma))
    )


@given(square_tensors(symmetric=True))
def test_is_symmetric_true(symmetric_tensor: Tensor) -> None:
    assert _is_symmetric(symmetric_tensor)


@given(square_tensors(symmetric=False))
def test_is_symmetric_false(non_symmetric_tensor: Tensor) -> None:
    assert not _is_symmetric(non_symmetric_tensor)


def test_uncertain_tensor_has_two_parameters() -> None:
    assert_equal(len(signature(UncertainTensor).parameters), 2)


def test_uncertain_tensor_has_parameter_values() -> None:
    assert "values" in signature(UncertainTensor).parameters


def test_uncertain_tensor_has_parameter_uncertainties() -> None:
    assert "uncertainties" in signature(UncertainTensor).parameters


def test_uncertain_tensor_parameter_values_is_tensor() -> None:
    assert issubclass(
        signature(UncertainTensor).parameters["values"].annotation, Tensor
    )


def test_uncertain_tensor_parameter_uncertainties_is_tensor() -> None:
    assert (
        signature(UncertainTensor).parameters["uncertainties"].annotation
        == Tensor | None
    )


@given(tensors())
def test_init_uncertain_tensor_allows_init_with_values_only(values: Tensor) -> None:
    UncertainTensor(values)


@given(tensors())
def test_init_uncertain_tensor_with_values_only_contains_values(values: Tensor) -> None:
    assert_close(UncertainTensor(values).values, values, equal_nan=True)


@given(tensors())
def test_init_uncertain_tensor_with_values_only_contains_values_at_index_zero(
    values: Tensor,
) -> None:
    assert_close(UncertainTensor(values)[0], values, equal_nan=True)


@given(uncertain_tensors())
def test_init_uncertain_tensor_contains_values(
    uncertain_tensor: UncertainTensor,
) -> None:
    assert_close(
        UncertainTensor(uncertain_tensor.values, uncertain_tensor.uncertainties).values,
        uncertain_tensor.values,
        equal_nan=True,
    )


@given(uncertain_tensors())
def test_init_uncertain_tensor_contains_values_at_index_zero(
    uncertain_tensor: UncertainTensor,
) -> None:
    assert_close(
        UncertainTensor(uncertain_tensor.values, uncertain_tensor.uncertainties)[0],
        uncertain_tensor.values,
        equal_nan=True,
    )


@given(uncertain_tensors())
def test_init_uncertain_tensor_contains_uncertainties(
    uncertain_tensor: UncertainTensor,
) -> None:
    assert_close(
        UncertainTensor(
            uncertain_tensor.values, uncertain_tensor.uncertainties
        ).uncertainties,
        uncertain_tensor.uncertainties,
        equal_nan=True,
    )


@given(uncertain_tensors())
def test_init_uncertain_tensor_contains_values_at_index_one(
    uncertain_tensor: UncertainTensor,
) -> None:
    assert_close(
        UncertainTensor(uncertain_tensor.values, uncertain_tensor.uncertainties)[1],
        uncertain_tensor.uncertainties,
        equal_nan=True,
    )


@given(uncertain_tensors())
def test_init_uncertain_tensor_with_named_parameter_value_only(
    uncertain_tensor: UncertainTensor,
) -> None:
    assert_close(
        UncertainTensor(values=uncertain_tensor.values),
        UncertainTensor(uncertain_tensor.values, None),
    )
