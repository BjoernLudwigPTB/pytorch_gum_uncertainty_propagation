"""Contains utilities to process measurement uncertainties"""

__all__ = [
    "cov_matrix_from_std_uncertainties",
    "is_positive_semi_definite",
    "is_symmetric",
    "UncertainTensor",
]

from typing import NamedTuple

import torch
from torch import Tensor


class UncertainTensor(NamedTuple):
    """A tuple of a tensor of values with a tensor of associated uncertainties"""

    values: Tensor
    uncertainties: Tensor | None = None


def cov_matrix_from_std_uncertainties(sigma: Tensor) -> Tensor:
    r"""Compute :math:`u(x) u^T(x)` given standard uncertainties:math:`u(x)`

    The resulting matrix will always be positive semi-definite.

    Parameters
    ----------
    sigma : Tensor of shape 1xn
        standard deviations

    Returns
    -------
    Tensor of shape n x n
        covariance matrix
    """
    cov_tensor = sigma * sigma.unsqueeze(1)
    assert is_symmetric(cov_tensor)
    assert is_positive_semi_definite(cov_tensor)
    return cov_tensor


def is_symmetric(matrix: Tensor) -> Tensor:
    """Returns True if matrix is symmetric while NaNs are considered equal

    Parameters
    ----------
    matrix : Tensor
        the matrix under test

    Returns
    -------
    Tensor[bool]
        True, if matrix is symmetric considering NaNs as equal, False otherwise

    Raises
    ------
    RuntimeError
        if matrix is not square
    """
    return torch.all(torch.isnan(matrix[~matrix.isclose(matrix.T)]))


def is_positive_semi_definite(matrix: Tensor) -> Tensor:
    """Returns True if tensor is positive semi-definite

    If there are :class:`torch.nan` or :class:`torch.inf` present in the matrix,
    unexpected behaviour can occur.

    Parameters
    ----------
    matrix : Tensor
        the matrix under test

    Returns
    -------
    Tensor[bool]
        True, if matrix is positive_semi_definite, False otherwise

    Raises
    ------
    RuntimeError
        if matrix is not square
    """
    if len(matrix) == 1:
        if matrix.shape[1] != 1:
            torch.linalg.eigvalsh(matrix)
        return matrix >= 0
    eigenvalues = torch.linalg.eigvalsh(matrix)
    return torch.all(
        torch.logical_or(
            eigenvalues >= 0,
            torch.isclose(eigenvalues, matrix.new_zeros(1), atol=1e-6),
        )
    )
