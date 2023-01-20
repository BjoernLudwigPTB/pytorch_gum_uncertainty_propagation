from inspect import signature

from pytorch_gum_uncertainty_propagation.examples import zema_dataset
from pytorch_gum_uncertainty_propagation.examples.zema_dataset import (
    convert_zema_std_uncertainties_into_synthetic_full_cov_matrices,
)
from pytorch_gum_uncertainty_propagation.uncertainties import (
    is_positive_semi_definite,
    UncertainTensor,
)


def test_zema_dataset_has_docstring() -> None:
    assert zema_dataset.__doc__ is not None


def test_zema_dataset_has_function_prepare_data() -> None:
    assert hasattr(
        zema_dataset, "convert_zema_std_uncertainties_into_synthetic_full_cov_matrices"
    )


def test_zema_dataset_all_contains_convert_std_uncerts_into_full_covs() -> None:
    assert (
        convert_zema_std_uncertainties_into_synthetic_full_cov_matrices.__name__
        in zema_dataset.__all__
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


def test_prepare_data_expects_parameter_size_scaler() -> None:
    assert (
        "size_scaler"
        in signature(
            convert_zema_std_uncertainties_into_synthetic_full_cov_matrices
        ).parameters
    )


def test_prepare_data_parameter_size_scaler_is_of_type_int() -> None:
    assert (
        signature(convert_zema_std_uncertainties_into_synthetic_full_cov_matrices)
        .parameters["size_scaler"]
        .annotation
        is int
    )


def test_prepare_data_parameter_size_scaler_default_is_one() -> None:
    assert (
        signature(convert_zema_std_uncertainties_into_synthetic_full_cov_matrices)
        .parameters["size_scaler"]
        .default
        == 1
    )


def test_prepare_data_expects_parameter_normalize() -> None:
    assert (
        "normalize"
        in signature(
            convert_zema_std_uncertainties_into_synthetic_full_cov_matrices
        ).parameters
    )


def test_prepare_data_parameter_normalize_is_of_type_bool() -> None:
    assert (
        signature(convert_zema_std_uncertainties_into_synthetic_full_cov_matrices)
        .parameters["normalize"]
        .annotation
        is bool
    )


def test_prepare_data_parameter_normalize_default_is_True() -> None:
    assert (
        signature(convert_zema_std_uncertainties_into_synthetic_full_cov_matrices)
        .parameters["normalize"]
        .default
        is True
    )


def test_prepare_data_expects_parameter_idx_start() -> None:
    assert (
        "idx_start"
        in signature(
            convert_zema_std_uncertainties_into_synthetic_full_cov_matrices
        ).parameters
    )


def test_prepare_data_parameter_idx_start_is_of_type_int() -> None:
    assert (
        signature(convert_zema_std_uncertainties_into_synthetic_full_cov_matrices)
        .parameters["idx_start"]
        .annotation
        is int
    )


def test_prepare_data_parameter_idx_start_default_is_zero() -> None:
    assert (
        signature(convert_zema_std_uncertainties_into_synthetic_full_cov_matrices)
        .parameters["idx_start"]
        .default
        == 0
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
        assert is_positive_semi_definite(cov_matrix)
