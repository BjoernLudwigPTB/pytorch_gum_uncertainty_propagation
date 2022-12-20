"""An API for accessing the data in the ZeMA remaining-useful life dataset"""

__all__ = [
    "ExtractionDataType",
    "provide_zema_samples",
    "LOCAL_ZEMA_DATASET_PATH",
    "ZEMA_DATASET_HASH",
    "ZEMA_DATASET_URL",
    "ZEMA_DATATYPES",
    "ZEMA_QUANTITIES",
]

from enum import Enum
from os.path import dirname, exists
from pathlib import Path

import h5py
import numpy as np
import torch
from h5py import Dataset, File, Group
from numpy._typing import NDArray
from pooch import retrieve

from pytorch_gum_uncertainty_propagation.uncertainties import (
    cov_matrix_from_std_uncertainties,
    UncertainTensor,
)

LOCAL_ZEMA_DATASET_PATH = Path(dirname(__file__), "datasets")
ZEMA_DATASET_HASH = (
    "sha256:fb0e80de4e8928ae8b859ad9668a1b6ea6310028a6690bb8d4c1abee31cb8833"
)
ZEMA_DATASET_URL = "https://zenodo.org/record/5185953/files/axis11_2kHz_ZeMA_PTB_SI.h5"
ZEMA_DATATYPES = ("qudt:standardUncertainty", "qudt:value")
ZEMA_QUANTITIES = (
    "Acceleration",
    "Active_Current",
    "Force",
    "Motor_Current",
    "Pressure",
    "Sound_Pressure",
    "Velocity",
)


class ExtractionDataType(Enum):
    """Identifiers of data types in ZeMA dataset

    Attributes
    ----------
    UNCERTAINTIES : str
        with value ``qudt:standardUncertainty``
    VALUES : str
        with value ``qudt:value``
    """

    UNCERTAINTIES = "qudt:standardUncertainty"
    VALUES = "qudt:value"


def provide_zema_samples(n_samples: int = 1) -> UncertainTensor:
    """Extracts requested number of samples of values with associated uncertainties

    The underlying dataset is the annotated "Sensor data set of one electromechanical
    cylinder at ZeMA testbed (ZeMA DAQ and Smart-Up Unit)" by Dorst et al. [Dorst2021]_.

    Parameters
    ----------
    n_samples : int
        number of samples each containing one reading from each of the eleven sensors
        with associated uncertainties

    Returns
    -------
    UncertainTensor
        The collection of samples of values with associated uncertainties
    """

    def _hdf5_part(hdf5_file: File, keys: list[str]) -> Group | Dataset:
        part = hdf5_file
        for key in keys:
            part = part[key]
        return part

    def _extract_sample_from_dataset(
        data_set: Dataset, ns_samples: tuple[slice, int]
    ) -> NDArray[np.double]:
        return np.expand_dims(np.array(data_set[ns_samples]), 1)

    def _append_to_extraction(
        append_to: NDArray[np.double], appendix: NDArray[np.double]
    ) -> NDArray[np.double]:
        return np.append(append_to, appendix, axis=1)

    dataset_full_path = retrieve(
        url=ZEMA_DATASET_URL,
        known_hash=ZEMA_DATASET_HASH,
        path=LOCAL_ZEMA_DATASET_PATH,
        progressbar=True,
    )
    assert exists(dataset_full_path)
    uncertainties = np.empty((n_samples, 0))
    values = np.empty((n_samples, 0))
    indices = np.s_[0:n_samples, 0]
    relevant_datasets = (
        ["ZeMA_DAQ", quantity, datatype]
        for quantity in ZEMA_QUANTITIES
        for datatype in ZEMA_DATATYPES
    )
    with h5py.File(dataset_full_path, "r") as h5f:
        for dataset in relevant_datasets:
            if ExtractionDataType.UNCERTAINTIES.value in dataset:
                extracted_data = uncertainties
                print(f"    Extract uncertainties from {dataset}")
            elif ExtractionDataType.VALUES.value in dataset:
                extracted_data = values
                print(f"    Extract values from {dataset}")
            else:
                extracted_data = None
            if extracted_data is not None:
                if len(_hdf5_part(h5f, dataset).shape) == 3:
                    for sensor in _hdf5_part(h5f, dataset):
                        extracted_data = _append_to_extraction(
                            extracted_data,
                            _extract_sample_from_dataset(sensor, indices),
                        )
                else:
                    extracted_data = _append_to_extraction(
                        extracted_data,
                        _extract_sample_from_dataset(
                            _hdf5_part(h5f, dataset),
                            indices,
                        ),
                    )
                if (
                    ExtractionDataType.UNCERTAINTIES.value
                    in _hdf5_part(h5f, dataset).name
                ):
                    uncertainties = extracted_data
                    print("    Uncertainties extracted")
                elif ExtractionDataType.VALUES.value in _hdf5_part(h5f, dataset).name:
                    values = extracted_data
                    print("    Values extracted")
    return UncertainTensor(torch.tensor(values), torch.tensor(uncertainties))


def convert_zema_std_uncertainties_into_synthetic_full_cov_matrices(
    n_samples: int = 1,
) -> UncertainTensor:
    """Prepare the ZeMA data for forward propagations in any PyTorch GUM-enabled network

    The main task is turning the standard uncertainties in the ZeMA dataset
    synthetically into full covariance matrices only for showcasing
    :class:`~pytorch_gum_uncertainty_propagation.modules.UncertainQuadLU`'s
    capabilities.
    """
    uncertain_values = provide_zema_samples(n_samples)
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
            sample, 0.5, 0.5, 0.5
        )
    return UncertainTensor(uncertain_values.values, result_uncertainties)
