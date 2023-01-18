"""Contains utilities to process measurement uncertainties"""

__all__ = ["cov_matrix_from_std_uncertainties", "UncertainTensor"]

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
    assert _is_symmetric(cov_tensor)
    assert _is_positive_semi_definite(cov_tensor)
    return cov_tensor


def _is_symmetric(matrix: Tensor) -> Tensor:
    """Returns True if matrix is symmetric"""
    return torch.all(torch.isnan(matrix[~matrix.isclose(matrix.T)]))


def _is_positive_semi_definite(tensor_under_test: Tensor) -> Tensor:
    """Returns True if tensor is positive semi-definite"""
    if len(tensor_under_test) == 1:
        return tensor_under_test >= 0
    eigenvalues = torch.linalg.eigvalsh(tensor_under_test)
    return torch.all(
        torch.logical_or(
            eigenvalues >= 0,
            torch.isclose(eigenvalues, tensor_under_test.new_zeros(1), atol=1e-6),
        )
    )
