from inspect import signature
from typing import Any, cast, Union

import numpy as np
import pytest
import torch
from hypothesis import assume, given, strategies as hst
from hypothesis.extra import numpy as hnp
from hypothesis.strategies import composite, DrawFn, SearchStrategy
from numpy import array, float64
from numpy._typing import NDArray
from numpy.testing import assert_equal
from torch import diag, isnan, tensor, Tensor
from torch.testing import assert_close  # type: ignore[attr-defined]

from pytorch_gum_uncertainty_propagation import uncertainties
from pytorch_gum_uncertainty_propagation.uncertainties import (
    _is_positive_semi_definite,
    _is_symmetric,
    _match_dimen_and_std_uncertainty_vec_len,
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
def input_for_cov_matrix_from_std_uncertainties(
    draw: DrawFn,
) -> SearchStrategy[tuple[Tensor, float, float, float]]:
    common_float_params = {
        "allow_subnormal": False,
        "allow_nan": False,
        "allow_infinity": False,
    }
    sigma_strategy = hst.floats(**common_float_params, min_value=0, max_value=10)
    array_strategy = hnp.arrays(
        float,
        shape=hnp.array_shapes(min_dims=1, max_dims=1, min_side=1, max_side=100),
        elements=hst.floats(**common_float_params, min_value=0.1, max_value=10),
    )
    sigma = tensor(draw(hst.one_of(sigma_strategy, array_strategy)))
    unit_params = {
        "exclude_min": True,
        "exclude_max": True,
    }
    rho = draw(
        hst.floats(**common_float_params, **unit_params, min_value=0.0, max_value=0.9)
    )
    random_unit_float_strategy = hst.floats(
        **common_float_params, **unit_params, min_value=0.1, max_value=1
    )
    phi = draw(random_unit_float_strategy)
    nu = draw(random_unit_float_strategy)
    return cast(
        SearchStrategy[tuple[Tensor, float, float, float]], (sigma, rho, phi, nu)
    )


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


def test_uncertainties_has_attribute_match_dimen_and_std_uncertainty_vec_len() -> None:
    assert hasattr(uncertainties, "_match_dimen_and_std_uncertainty_vec_len")


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


@given(
    hnp.arrays(
        float, hnp.array_shapes(min_dims=1, max_dims=10, min_side=1, max_side=1)
    ),
    hst.one_of(hst.just(None), hst.integers(min_value=1, max_value=10)),
)
def test_ensure_matching_dimension_and_std_uncertainty_vector_is_tuple(
    sigma: NDArray[Any], dimen: Union[int, None]
) -> None:
    sigma_tensor = tensor(sigma)
    assert isinstance(
        _match_dimen_and_std_uncertainty_vec_len(sigma_tensor, dimen),
        tuple,
    )


@given(
    hnp.arrays(
        float, hnp.array_shapes(min_dims=1, max_dims=10, min_side=1, max_side=1)
    ),
    hst.one_of(hst.just(None), hst.integers(min_value=1, max_value=10)),
)
def test_ensure_matching_dimension_and_std_uncertainty_vector_first_is_ndarray(
    sigma: NDArray[float64], dimen: Union[int, None]
) -> None:
    sigma_tensor = tensor(sigma)
    assert isinstance(
        _match_dimen_and_std_uncertainty_vec_len(sigma_tensor, dimen)[0],
        Tensor,
    )


@given(
    hnp.arrays(
        float, hnp.array_shapes(min_dims=1, max_dims=10, min_side=1, max_side=1)
    ),
    hst.one_of(hst.just(None), hst.integers(min_value=1, max_value=10)),
)
def test_ensure_matching_dimension_and_std_uncertainty_vector_second_is_int(
    sigma: NDArray[float64], dimen: Union[int, None]
) -> None:
    sigma_tensor = tensor(sigma)
    assert isinstance(
        _match_dimen_and_std_uncertainty_vec_len(sigma_tensor, dimen)[1],
        int,
    )


@given(
    hst.floats(allow_nan=False),
    hst.one_of(hst.just(None), hst.integers(min_value=1, max_value=10)),
)
def test_ensure_matching_dimension_and_std_uncertainty_vector_first_is_ndarray_of_sigma(
    sigma: float, dimen: int
) -> None:
    sigma_tensor = tensor([sigma])
    sigma_result, _ = _match_dimen_and_std_uncertainty_vec_len(sigma_tensor, dimen)
    for sigma_i in sigma_result:
        assert sigma_i == sigma


@given(
    hnp.arrays(
        float, hnp.array_shapes(min_dims=1, max_dims=10, min_side=1, max_side=1)
    ),
    hst.one_of(hst.just(None), hst.integers(min_value=1, max_value=10)),
)
def test_ensure_matching_dimension_and_std_uncertainty_vector_dimension_matches_ndarray(
    sigma: NDArray[float64], dimen: int
) -> None:
    sigma_tensor = tensor(sigma)
    result = _match_dimen_and_std_uncertainty_vec_len(sigma_tensor, dimen)
    assert len(result[0]) == result[1]


@given(
    hnp.arrays(
        float, hnp.array_shapes(min_dims=1, max_dims=10, min_side=1, max_side=1)
    ),
    hst.integers(min_value=1, max_value=10),
)
def test_ensure_matching_dimension_and_std_uncertainty_vector_dimensions(
    sigma: NDArray[float64], dimen: int
) -> None:
    sigma_tensor = tensor(sigma)
    assert _match_dimen_and_std_uncertainty_vec_len(sigma_tensor, dimen)[1] == dimen


@given(input_for_cov_matrix_from_std_uncertainties())
def test_cov_matrix_from_std_uncertainties(
    sigma_rho_phi_nu: tuple[Tensor, float, float, float]
) -> None:
    assert isinstance(cov_matrix_from_std_uncertainties(*sigma_rho_phi_nu), Tensor)


@given(square_tensors(symmetric=True))
def test_is_symmetric_true(symmetric_tensor: Tensor) -> None:
    assert _is_symmetric(symmetric_tensor)


@given(square_tensors(symmetric=False))
def test_is_symmetric_false(non_symmetric_tensor: Tensor) -> None:
    assert not _is_symmetric(non_symmetric_tensor)


@given(input_for_cov_matrix_from_std_uncertainties())
def test_cov_matrix_from_std_uncertainties_raises_exception(
    sigma_rho_phi_nu: tuple[Tensor, float, float, float]
) -> None:
    assume(bool(len(sigma_rho_phi_nu[0].shape) and len(sigma_rho_phi_nu[0]) >= 2))
    with pytest.raises(AssertionError):
        cov_matrix_from_std_uncertainties(
            sigma_rho_phi_nu[0], 2.0, *sigma_rho_phi_nu[2:]
        )


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
