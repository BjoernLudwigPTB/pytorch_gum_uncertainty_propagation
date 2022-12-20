from inspect import signature

import pytest
from hypothesis import given, settings, strategies as hst

from pytorch_gum_uncertainty_propagation.examples import (
    uncertain_quadlu_propagate,
)
from pytorch_gum_uncertainty_propagation.examples.uncertain_quadlu_propagate import (
    _construct_partition,
    assemble_pipeline,
    instantiate_uncertain_quadlu_mlp,
    prepare_data,
)
from pytorch_gum_uncertainty_propagation.modules import UncertainQuadLUMLP
from pytorch_gum_uncertainty_propagation.uncertainties import (
    _is_positive_semi_definite,
    UncertainTensor,
)


def test_uncertain_quadlu_propagate_has_docstring() -> None:
    assert uncertain_quadlu_propagate.__doc__ is not None


def test_uncertain_quadlu_propagate_has_function_prepare_data() -> None:
    assert hasattr(uncertain_quadlu_propagate, "prepare_data")


def test_uncertain_quadlu_propagate_has_function_propagate() -> None:
    assert hasattr(uncertain_quadlu_propagate, "assemble_pipeline")


def test_uncertain_quadlu_propagate_all_contains_prepare_data() -> None:
    assert assemble_pipeline.__name__ in uncertain_quadlu_propagate.__all__


def test_uncertain_quadlu_propagate_has_instantiate_uncertain_quadlu_mlp() -> None:
    assert hasattr(uncertain_quadlu_propagate, "instantiate_uncertain_quadlu_mlp")


def test_prepare_data_has_docstring() -> None:
    assert prepare_data.__doc__ is not None


def test_prepare_data_expects_parameter_n_samples() -> None:
    assert "n_samples" in signature(prepare_data).parameters


def test_prepare_data_parameter_n_samples_is_of_type_int() -> None:
    assert signature(prepare_data).parameters["n_samples"].annotation is int


def test_prepare_data_parameter_n_samples_default_is_one() -> None:
    assert signature(prepare_data).parameters["n_samples"].default == 1


def test_propagate_has_docstring() -> None:
    assert assemble_pipeline.__doc__ is not None


def test_instantiate_uncertain_quadlu_mlp_has_docstring() -> None:
    assert instantiate_uncertain_quadlu_mlp.__doc__ is not None


def test_instantiate_uncertain_quadlu_mlp_has_in_features_parameter() -> None:
    assert "in_features" in signature(instantiate_uncertain_quadlu_mlp).parameters


def test_instantiate_uncertain_quadlu_mlp_in_features_parameter_is_int() -> None:
    assert (
        signature(instantiate_uncertain_quadlu_mlp).parameters["in_features"].annotation
        is int
    )


def test_instantiate_uncertain_quadlu_mlp_has_out_features_parameter() -> None:
    assert "out_features" in signature(instantiate_uncertain_quadlu_mlp).parameters


def test_instantiate_uncertain_quadlu_mlp_out_features_parameter_is_int_list() -> None:
    assert (
        signature(instantiate_uncertain_quadlu_mlp)
        .parameters["out_features"]
        .annotation
        == list[int]
    )


def test_prepare_data_provides_uncertain_tensor() -> None:
    assert issubclass(signature(prepare_data).return_annotation, UncertainTensor)


def test_prepare_data_actually_returns_uncertain_tensor() -> None:
    assert isinstance(prepare_data(), UncertainTensor)


def test_prepare_data_returns_full_covariance() -> None:
    uncertainties = prepare_data().uncertainties
    assert uncertainties is not None
    shape = uncertainties.shape
    assert len(shape) == 3 and shape[-1] == shape[-2]


def test_prepare_data_returns_positive_semi_definite_covariance() -> None:
    uncertainties = prepare_data().uncertainties
    assert uncertainties is not None
    for cov_matrix in uncertainties:
        assert _is_positive_semi_definite(cov_matrix)


@given(
    hst.integers(min_value=1, max_value=10),
    hst.lists(hst.integers(min_value=1, max_value=10), min_size=1, max_size=10),
)
def test_instantiate_uncertain_quadlu_mlp_returns_uncertain_quadlu_mlp(
    in_features: int, out_features: list[int]
) -> None:
    assert isinstance(
        instantiate_uncertain_quadlu_mlp(in_features, out_features), UncertainQuadLUMLP
    )


def test_propagate_expects_parameter_uncertain_values() -> None:
    assert "uncertain_values" in signature(assemble_pipeline).parameters


def test_propagate_expects_uncertain_values_of_type_uncertain_values() -> None:
    assert (
        signature(assemble_pipeline).parameters["uncertain_values"].annotation
        == UncertainTensor | None
    )


def test_propagate_expects_uncertain_values_default_is_None() -> None:
    assert signature(assemble_pipeline).parameters["uncertain_values"].default is None


def test_construct_partition_exists() -> None:
    assert hasattr(uncertain_quadlu_propagate, "_construct_partition")


def test_construct_partition_has_docstring() -> None:
    assert _construct_partition.__doc__ is not None


def test_construct_partition_has_parameter_in_features() -> None:
    assert "in_features" in signature(_construct_partition).parameters


def test_construct_partition_parameter_in_features_is_of_type_int() -> None:
    assert signature(_construct_partition).parameters["in_features"].annotation is int


def test_construct_partition_parameter_states_to_return_int_list() -> None:
    assert signature(_construct_partition).return_annotation == list[int]


@given(hst.integers(min_value=1, max_value=10))
def test_construct_partition_actually_returns_int_list(in_features: int) -> None:
    assert isinstance(_construct_partition(in_features), list)


@given(hst.integers(min_value=1, max_value=100))
def test_construct_partition_returns_non_empty_list(in_features: int) -> None:
    assert len(_construct_partition(in_features))


def test_construct_partition_returns_correct_small_example() -> None:
    assert _construct_partition(10) == [10, 7, 5, 3, 2]


@given(hst.integers(min_value=1, max_value=100))
def test_construct_partition_is_descending(in_features: int) -> None:
    partition = _construct_partition(in_features)
    assert partition[:] > partition[1:]


def test_construct_partition_returns_correct_large_example() -> None:
    assert _construct_partition(100) == [100, 75, 56, 42, 31]


@pytest.mark.webtest
@given(hst.integers(min_value=1, max_value=10))
@settings(deadline=None)
def test_propagate_actually_runs(n_samples: int) -> None:
    assemble_pipeline(n_samples)


def test_propagate_expects_parameter_n_samples() -> None:
    assert "n_samples" in signature(assemble_pipeline).parameters


def test_propagate_parameter_n_samples_is_of_type_int() -> None:
    assert signature(assemble_pipeline).parameters["n_samples"].annotation is int


def test_propagate_parameter_n_samples_default_is_one() -> None:
    assert signature(assemble_pipeline).parameters["n_samples"].default == 1
