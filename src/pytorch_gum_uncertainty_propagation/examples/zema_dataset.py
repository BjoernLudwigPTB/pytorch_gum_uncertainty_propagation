"""An API for accessing the data in the ZeMA remaining-useful life dataset"""

__all__ = ["convert_zema_std_uncertainties_into_synthetic_full_cov_matrices"]

import torch
from zema_emc_annotated.data_types import SampleSize
from zema_emc_annotated.dataset import ZeMASamples  # type: ignore[import]

from pytorch_gum_uncertainty_propagation.uncertainties import (
    cov_matrix_from_std_uncertainties,
    UncertainTensor,
)


def convert_zema_std_uncertainties_into_synthetic_full_cov_matrices(
    n_samples: int = 1,
    size_scaler: int = 1,
    normalize: bool = True,
    idx_start: int = 0,
) -> UncertainTensor:
    """Prepare the ZeMA data for forward propagations in any PyTorch GUM-enabled network

    The main task is turning the standard uncertainties in the ZeMA dataset [Dorst2021]_
    synthetically into full covariance matrices only for showcasing the capabilities
    of the GUM-enabled :doc:`pytorch_gum_uncertainty_propagation.modules`.
    """
    uncertain_array = ZeMASamples(
        SampleSize(idx_start, n_samples, size_scaler), normalize, True
    )
    uncertain_values = UncertainTensor(
        values=torch.from_numpy(uncertain_array.values),
        uncertainties=torch.from_numpy(uncertain_array.uncertainties),
    )
    assert uncertain_values.uncertainties is not None
    result_uncertainties = torch.empty(
        (
            len(uncertain_values.uncertainties),
            uncertain_values.uncertainties.shape[1],
            uncertain_values.uncertainties.shape[1],
        )
    )
    for sample_idx, sample in enumerate(uncertain_values.uncertainties):
        result_uncertainties[sample_idx, ...] = cov_matrix_from_std_uncertainties(
            sample
        )
    return UncertainTensor(uncertain_values.values, result_uncertainties)
