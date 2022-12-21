import os
from inspect import signature
from pathlib import Path

import pytest
import torch
from hypothesis import given, settings, strategies as hst

from pytorch_gum_uncertainty_propagation import zema_dataset
from pytorch_gum_uncertainty_propagation.uncertainties import (
    _is_positive_semi_definite,
    UncertainTensor,
)
from pytorch_gum_uncertainty_propagation.zema_dataset import (
    _cache_path,
    _check_and_load_cache,
    _store_cache,
    convert_zema_std_uncertainties_into_synthetic_full_cov_matrices,
    ExtractionDataType,
    LOCAL_ZEMA_DATASET_PATH,
    provide_zema_samples,
    ZEMA_DATASET_HASH,
    ZEMA_DATASET_URL,
    ZEMA_DATATYPES,
    ZEMA_QUANTITIES,
)
from .conftest import uncertain_tensors


def test_zema_dataset_has_docstring() -> None:
    assert zema_dataset.__doc__ is not None


def test_zema_dataset_has_enum_extraction_data() -> None:
    assert hasattr(zema_dataset, "ExtractionDataType")


def test_extraction_data_enum_has_docstring_with_values() -> None:
    assert ExtractionDataType.__doc__ is not None
    assert "VALUES" in ExtractionDataType.__doc__


def test_extraction_data_enum_has_docstring_with_uncertainties() -> None:
    assert ExtractionDataType.__doc__ is not None
    assert "UNCERTAINTIES" in ExtractionDataType.__doc__


def test_zema_dataset_extraction_data_contains_key_for_uncertainties() -> None:
    assert "qudt:standardUncertainty" in ExtractionDataType._value2member_map_


def test_zema_dataset_extraction_data_contains_key_for_values() -> None:
    assert "qudt:value" in ExtractionDataType._value2member_map_


def test_zema_dataset_all_contains_extraction_data() -> None:
    assert ExtractionDataType.__name__ in zema_dataset.__all__


def test_zema_dataset_has_constant_datatypes() -> None:
    assert hasattr(zema_dataset, "ZEMA_DATATYPES")


def test_zema_dataset_constant_datatypes_is_tuple() -> None:
    assert isinstance(ZEMA_DATATYPES, tuple)


def test_zema_dataset_constant_datatypes_contains_uncertainties() -> None:
    assert "qudt:standardUncertainty" in ZEMA_DATATYPES


def test_zema_dataset_constant_datatypes_contains_for_values() -> None:
    assert "qudt:value" in ZEMA_DATATYPES


def test_zema_dataset_all_contains_constant_datatypes() -> None:
    assert "ZEMA_DATATYPES" in zema_dataset.__all__


def test_zema_dataset_has_constant_quantities() -> None:
    assert hasattr(zema_dataset, "ZEMA_QUANTITIES")


def test_zema_dataset_constant_quantities_is_tuple() -> None:
    assert isinstance(ZEMA_QUANTITIES, tuple)


def test_zema_dataset_constant_quantities_contains_acceleration() -> None:
    assert "Acceleration" in ZEMA_QUANTITIES


def test_zema_dataset_constant_quantities_contains_active_current() -> None:
    assert "Active_Current" in ZEMA_QUANTITIES


def test_zema_dataset_constant_quantities_contains_force() -> None:
    assert "Force" in ZEMA_QUANTITIES


def test_zema_dataset_constant_quantities_contains_motor_current() -> None:
    assert "Motor_Current" in ZEMA_QUANTITIES


def test_zema_dataset_constant_quantities_contains_pressure() -> None:
    assert "Pressure" in ZEMA_QUANTITIES


def test_zema_dataset_constant_quantities_contains_sound_pressure() -> None:
    assert "Sound_Pressure" in ZEMA_QUANTITIES


def test_zema_dataset_constant_quantities_contains_velocity() -> None:
    assert "Velocity" in ZEMA_QUANTITIES


def test_zema_dataset_all_contains_constant_quantities() -> None:
    assert "ZEMA_QUANTITIES" in zema_dataset.__all__


def test_zema_dataset_has_attribute_local_zema_dataset_path() -> None:
    assert hasattr(zema_dataset, "LOCAL_ZEMA_DATASET_PATH")


def test_zema_dataset_attribute_local_zema_dataset_path_is_path() -> None:
    assert isinstance(LOCAL_ZEMA_DATASET_PATH, Path)


def test_zema_dataset_attribute_local_zema_dataset_path_in_all() -> None:
    assert "LOCAL_ZEMA_DATASET_PATH" in zema_dataset.__all__


def test_zema_dataset_has_attribute_zema_dataset_url() -> None:
    assert hasattr(zema_dataset, "ZEMA_DATASET_URL")


def test_zema_dataset_attribute_zema_dataset_url_is_string() -> None:
    assert isinstance(ZEMA_DATASET_URL, str)


def test_zema_dataset_attribute_zema_dataset_url_in_all() -> None:
    assert "ZEMA_DATASET_URL" in zema_dataset.__all__


def test_zema_dataset_has_attribute_zema_dataset_hash() -> None:
    assert hasattr(zema_dataset, "ZEMA_DATASET_HASH")


def test_zema_dataset_attribute_zema_dataset_hash() -> None:
    assert isinstance(ZEMA_DATASET_HASH, str)


def test_zema_dataset_attribute_zema_dataset_hash_in_all() -> None:
    assert "ZEMA_DATASET_HASH" in zema_dataset.__all__


def test_zema_dataset_has_attribute_extract_samples() -> None:
    assert hasattr(zema_dataset, "provide_zema_samples")


def test_zema_dataset_extract_samples_is_callable() -> None:
    assert callable(provide_zema_samples)


def test_zema_dataset_all_contains_extract_samples() -> None:
    assert provide_zema_samples.__name__ in zema_dataset.__all__


def test_extract_samples_has_docstring() -> None:
    assert provide_zema_samples.__doc__ is not None


def test_zema_dataset_has_attribute_check_and_load_cache() -> None:
    assert hasattr(zema_dataset, "_check_and_load_cache")


def test_zema_dataset_check_and_load_cache_is_callable() -> None:
    assert callable(_check_and_load_cache)


def test_check_and_load_cache_has_docstring() -> None:
    assert _check_and_load_cache.__doc__ is not None


def test_check_and_load_cache_expects_parameter_n_samples() -> None:
    assert "n_samples" in signature(_check_and_load_cache).parameters


def test_check_and_load_cache_expects_parameter_n_samples_as_int() -> None:
    assert signature(_check_and_load_cache).parameters["n_samples"].annotation is int


def test_zema_dataset_has_attribute_cache_path() -> None:
    assert hasattr(zema_dataset, "_cache_path")


def test_zema_dataset_cache_path_is_callable() -> None:
    assert callable(_cache_path)


def test_cache_path_has_docstring() -> None:
    assert _cache_path.__doc__ is not None


def test_cache_path_expects_parameter_n_samples() -> None:
    assert "n_samples" in signature(_cache_path).parameters


def test_cache_path_expects_parameter_n_samples_as_int() -> None:
    assert signature(_cache_path).parameters["n_samples"].annotation is int


@given(hst.integers())
def test_cache_path_actually_returns_path(integer: int) -> None:
    assert isinstance(_cache_path(integer), Path)


def test_zema_dataset_has_attribute_store_cache() -> None:
    assert hasattr(zema_dataset, "_store_cache")


def test_zema_dataset_store_cache_is_callable() -> None:
    assert callable(_store_cache)


def test_store_cache_has_docstring() -> None:
    assert _store_cache.__doc__ is not None


def test_store_cache_expects_parameter_uncertain_values() -> None:
    assert "uncertain_values" in signature(_store_cache).parameters


@given(uncertain_tensors(length=11))
def test_store_cache_runs_for_random_uncertain_values(
    uncertain_tensor: UncertainTensor,
) -> None:
    _store_cache(uncertain_tensor)
    assert os.path.exists(_cache_path(11))


@given(hst.integers())
def test_check_and_load_cache_runs_for_random_uncertain_values_and_returns(
    integer: int,
) -> None:
    result = _check_and_load_cache(integer)
    assert result is None or isinstance(result, UncertainTensor)


@given(uncertain_tensors(length=12))
def test_check_and_load_cache_returns_something_for_existing_file(
    uncertain_tensor: UncertainTensor,
) -> None:
    _store_cache(uncertain_tensor)
    assert _check_and_load_cache(12) is not None


def test_store_cache_expects_parameter_uncertain_values_as_uncertain_tensor() -> None:
    assert (
        signature(_store_cache).parameters["uncertain_values"].annotation
        is UncertainTensor
    )


def test_cache_path_expects_stats_to_return_path() -> None:
    assert signature(_cache_path).return_annotation is Path


def test_zema_dataset_extract_samples_expects_parameter_n_samples() -> None:
    assert "n_samples" in signature(provide_zema_samples).parameters


def test_zema_dataset_extract_samples_expects_parameter_n_samples_as_int() -> None:
    assert signature(provide_zema_samples).parameters["n_samples"].annotation is int


def test_zema_dataset_extract_samples_parameter_n_samples_default_is_one() -> None:
    assert signature(provide_zema_samples).parameters["n_samples"].default == 1


def test_zema_dataset_extract_samples_states_to_return_uncertain_tensor() -> None:
    assert signature(provide_zema_samples).return_annotation is UncertainTensor


@pytest.mark.webtest
@given(hst.integers(min_value=1, max_value=10))
@settings(deadline=None)
def test_extract_samples_actually_returns_uncertain_tensor(n_samples: int) -> None:
    assert isinstance(provide_zema_samples(n_samples), UncertainTensor)


@pytest.mark.webtest
@given(hst.integers(min_value=1, max_value=10))
@settings(deadline=None)
def test_extract_samples_actually_returns_uncertain_tensor_with_n_samples_values(
    n_samples: int,
) -> None:
    assert len(provide_zema_samples(n_samples).values) == n_samples


@pytest.mark.webtest
@given(hst.integers(min_value=1, max_value=10))
@settings(deadline=None)
def test_extract_samples_actually_returns_uncertain_tensor_with_n_samples_uncertainties(
    n_samples: int,
) -> None:
    result_uncertainties = provide_zema_samples(n_samples).uncertainties
    assert result_uncertainties is not None
    assert len(result_uncertainties) == n_samples


@pytest.mark.webtest
@given(hst.integers(min_value=1, max_value=10))
@settings(deadline=None)
def test_extract_samples_returns_values_of_eleven_sensors(
    n_samples: int,
) -> None:
    assert provide_zema_samples(n_samples).values.shape[1] == 11


@pytest.mark.webtest
@given(hst.integers(min_value=1, max_value=10))
@settings(deadline=None)
def test_extract_samples_returns_uncertainties_of_eleven_sensors(
    n_samples: int,
) -> None:
    result_uncertainties = provide_zema_samples(n_samples).uncertainties
    assert result_uncertainties is not None
    assert result_uncertainties.shape[1] == 11


@pytest.mark.webtest
@given(hst.integers(min_value=1, max_value=10))
@settings(deadline=None)
def test_extract_samples_returns_values_and_uncertainties_which_are_not_similar(
    n_samples: int,
) -> None:
    result = provide_zema_samples(n_samples)
    assert not torch.all(result.values == result.uncertainties)


def test_zema_dataset_has_function_prepare_data() -> None:
    assert hasattr(
        zema_dataset, "convert_zema_std_uncertainties_into_synthetic_full_cov_matrices"
    )


def test_prepare_data_has_docstring() -> None:
    assert (
        convert_zema_std_uncertainties_into_synthetic_full_cov_matrices.__doc__
        is not None
    )


def test_prepare_data_expects_parameter_n_samples() -> None:
    assert (
        "n_samples"
        in signature(
            convert_zema_std_uncertainties_into_synthetic_full_cov_matrices
        ).parameters
    )


def test_prepare_data_parameter_n_samples_is_of_type_int() -> None:
    assert (
        signature(convert_zema_std_uncertainties_into_synthetic_full_cov_matrices)
        .parameters["n_samples"]
        .annotation
        is int
    )


def test_prepare_data_parameter_n_samples_default_is_one() -> None:
    assert (
        signature(convert_zema_std_uncertainties_into_synthetic_full_cov_matrices)
        .parameters["n_samples"]
        .default
        == 1
    )


def test_prepare_data_provides_uncertain_tensor() -> None:
    assert issubclass(
        signature(
            convert_zema_std_uncertainties_into_synthetic_full_cov_matrices
        ).return_annotation,
        UncertainTensor,
    )


def test_prepare_data_actually_returns_uncertain_tensor() -> None:
    assert isinstance(
        convert_zema_std_uncertainties_into_synthetic_full_cov_matrices(),
        UncertainTensor,
    )


def test_prepare_data_returns_full_covariance() -> None:
    uncertainties = (
        convert_zema_std_uncertainties_into_synthetic_full_cov_matrices().uncertainties
    )
    assert uncertainties is not None
    shape = uncertainties.shape
    assert len(shape) == 3 and shape[-1] == shape[-2]


def test_prepare_data_returns_positive_semi_definite_covariance() -> None:
    uncertainties = (
        convert_zema_std_uncertainties_into_synthetic_full_cov_matrices().uncertainties
    )
    assert uncertainties is not None
    for cov_matrix in uncertainties:
        assert _is_positive_semi_definite(cov_matrix)
