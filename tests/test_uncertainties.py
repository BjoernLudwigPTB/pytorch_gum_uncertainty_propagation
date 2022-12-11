from typing import Any, cast, Optional, Union

import numpy as np
import pytest
import torch
from hypothesis import assume, given, strategies as hst
from hypothesis.extra import numpy as hnp
from hypothesis.strategies import composite, DrawFn, SearchStrategy
from numpy import array, float64
from numpy._typing import NDArray
from numpy.linalg import eigvals
from torch import diag, isnan, tensor, Tensor

from gum_compliant_neural_network_uncertainty_propagation.uncertainties import (
    _is_positive_semi_definite,
    _is_symmetric,
    _match_dimen_and_std_uncertainty_vec_len,
    cov_matrix_from_std_uncertainties,
)


@composite
def square_tensors(
    draw: DrawFn, dimen: Optional[int] = None, symmetric: Optional[bool] = False
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
                if (
                    not (
                        isnan(result_tensor[i_row, i_column])
                        and isnan(result_tensor[i_column, i_row])
                    )
                    and result_tensor[i_row, i_column] != result_tensor[i_column, i_row]
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


@given(hnp.arrays(float, hnp.array_shapes()))
def test_is_semi_positive_definite_usual_call(matrix: NDArray[Any]) -> None:
    tmp_tensor = tensor(matrix)
    assert isinstance(_is_positive_semi_definite(tmp_tensor), bool) or isinstance(
        _is_positive_semi_definite(tmp_tensor).item(), bool
    )


@given(hst.integers(min_value=1, max_value=10))
def test_is_semi_positive_definite_against_zero(length: int) -> None:
    assert _is_positive_semi_definite(tensor(np.zeros((length, length))))


def test_is_semi_positive_definite_for_single_instance_true() -> None:
    A = tensor([[5, 2, 1], [2, 4, 2], [1, 2, 3]])
    smallest_eigenvalue = min(eigvals(A))
    assert _is_positive_semi_definite(A) == tensor(bool(smallest_eigenvalue > 0))


def test_is_semi_positive_definite_for_single_instance_false() -> None:
    A = tensor(array([[1, 2, 1], [2, 2, 2], [1, 2, 3]]))
    smallest_eigenvalue = min(eigvals(A))
    assert _is_positive_semi_definite(A) == tensor(bool(smallest_eigenvalue > 0))


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
