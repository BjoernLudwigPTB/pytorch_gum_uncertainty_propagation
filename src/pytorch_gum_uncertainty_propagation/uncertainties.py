"""Contains utilities to process measurement uncertainties"""

__all__ = ["cov_matrix_from_std_uncertainties", "UncertainTensor"]

from typing import NamedTuple

import torch
from torch import Tensor


class UncertainTensor(NamedTuple):
    """A tuple of a tensor of values with a tensor of associated uncertainties"""

    values: Tensor
    uncertainties: Tensor | None = None


def cov_matrix_from_std_uncertainties(
    sigma: Tensor, rho: float, phi: float, nju: float, dimen: int | None = None
) -> Tensor:
    r"""Set up a covariance matrix from given standard uncertainties

    The matrix has the form :math:`V_{ij} = \sigma_i \sigma_j \rho^{\phi * i^\nu`

    `V = cov_matrix_from_std_uncertainties(sigma, rho, phi, nu, dimen)`
    constructs a covariance matrix as specified in Dierl et al., J. Opt. Soc.
    Am. A 33, 1370 (2016) with the column dimension of :math:'\sigma'. In case
    the dimension of :math:`\sigma` does not match `dimen`, its elements are either
    cut off or repeated until the desired length is reached. The main idea is to assume
    higher correlation of values for closer adjacency.

    Parameters
    ----------
    sigma : Tensor of shape 1xn
        standard deviation
    rho : float
        scale how many adjacent elements of the covariant structure are related
    phi : float
        scale how many adjacent elements of the covariant structure are related
    nju : float
        scale hwo many adjacent elements of the covariant structure are related
    dimen : float
        number of rows and columns of the covariance matrix

    Returns
    -------
    Tensor of shape n x n
        covariance matrix

    copyright PTB 2022, B. Ludwig, M. Dierl, D. Oberländer
    """
    sigma, dimen = _match_dimen_and_std_uncertainty_vec_len(sigma, dimen)
    corr_factors = torch.eye(dimen)
    for row in range(1, dimen):
        torch_diagonal = torch.tensor(rho ** (phi * row**nju)).expand(dimen - row)
        corr_factors += torch.diag_embed(torch_diagonal, offset=row)
        corr_factors += torch.diag_embed(torch_diagonal, offset=-row)
    cov_tensor = torch.mul(sigma, sigma.reshape(-1, 1)) * corr_factors
    assert _is_symmetric(cov_tensor)
    assert _is_positive_semi_definite(cov_tensor)
    return cov_tensor


def _is_symmetric(matrix: Tensor) -> Tensor:
    """Returns True if matrix is symmetric"""
    return torch.all(torch.isnan(matrix[~matrix.eq(matrix.T)]))


def _is_positive_semi_definite(tensor_under_test: Tensor) -> Tensor:
    """Returns True if tensor is positive semi-definite"""
    try:
        torch.linalg.cholesky(tensor_under_test.to(torch.float64))
        return torch.tensor(True)
    except RuntimeError:
        return torch.all(
            torch.isclose(tensor_under_test, tensor_under_test.new_zeros(1))
        )


def _match_dimen_and_std_uncertainty_vec_len(
    std_uncertainty: Tensor, desired_length: int | None = None
) -> tuple[Tensor, int]:
    """Adapt dimension specifier or vector of standard uncertainties"""
    if desired_length is None:
        desired_length = 1 if std_uncertainty.dim() == 0 else len(std_uncertainty)
    else:
        if desired_length > len(std_uncertainty):
            std_uncertainty = torch.repeat_interleave(
                std_uncertainty, int(desired_length / len(std_uncertainty))
            )
        std_uncertainty = std_uncertainty[:desired_length]
    return std_uncertainty, desired_length
