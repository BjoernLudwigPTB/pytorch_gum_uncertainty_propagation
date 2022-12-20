from inspect import signature
from typing import Any

import pytest
from hypothesis import given, settings, strategies as hst
from torch.autograd.profiler import profile

from pytorch_gum_uncertainty_propagation.examples import (
    uncertain_quadlu_propagate,
)
from pytorch_gum_uncertainty_propagation.examples.uncertain_quadlu_propagate import (
    _construct_partition,
    _instantiate_uncertain_quadlu_mlp,
    assemble_pipeline,
)
from pytorch_gum_uncertainty_propagation.modules import UncertainQuadLUMLP
from pytorch_gum_uncertainty_propagation.uncertainties import (
    UncertainTensor,
)


def test_uncertain_quadlu_propagate_has_docstring() -> None:
    assert uncertain_quadlu_propagate.__doc__ is not None


def test_uncertain_quadlu_propagate_has_function_assemble_pipeline() -> None:
    assert hasattr(uncertain_quadlu_propagate, "assemble_pipeline")


def test_uncertain_quadlu_propagate_all_contains_assemble_pipeline() -> None:
    assert assemble_pipeline.__name__ in uncertain_quadlu_propagate.__all__


def test_uncertain_quadlu_propagate_has_instantiate_uncertain_quadlu_mlp() -> None:
    assert hasattr(uncertain_quadlu_propagate, "_instantiate_uncertain_quadlu_mlp")


def test_assemble_pipeline_has_docstring() -> None:
    assert assemble_pipeline.__doc__ is not None


def test_instantiate_uncertain_quadlu_mlp_has_docstring() -> None:
    assert _instantiate_uncertain_quadlu_mlp.__doc__ is not None


def test_instantiate_uncertain_quadlu_mlp_has_in_features_parameter() -> None:
    assert "in_features" in signature(_instantiate_uncertain_quadlu_mlp).parameters


def test_instantiate_uncertain_quadlu_mlp_in_features_parameter_is_int() -> None:
    assert (
        signature(_instantiate_uncertain_quadlu_mlp)
        .parameters["in_features"]
        .annotation
        is int
    )


def test_instantiate_uncertain_quadlu_mlp_has_out_features_parameter() -> None:
    assert "out_features" in signature(_instantiate_uncertain_quadlu_mlp).parameters


def test_instantiate_uncertain_quadlu_mlp_out_features_parameter_is_int_list() -> None:
    assert (
        signature(_instantiate_uncertain_quadlu_mlp)
        .parameters["out_features"]
        .annotation
        == list[int]
    )


@given(
    hst.integers(min_value=1, max_value=10),
    hst.lists(hst.integers(min_value=1, max_value=10), min_size=1, max_size=10),
)
def test_instantiate_uncertain_quadlu_mlp_returns_uncertain_quadlu_mlp(
    in_features: int, out_features: list[int]
) -> None:
    assert isinstance(
        _instantiate_uncertain_quadlu_mlp(in_features, out_features), UncertainQuadLUMLP
    )


def test_assemble_pipeline_expects_parameter_uncertain_values() -> None:
    assert "uncertain_values" in signature(assemble_pipeline).parameters


def test_assemble_pipeline_expects_uncertain_values_of_type_uncertain_values() -> None:
    assert (
        signature(assemble_pipeline).parameters["uncertain_values"].annotation
        == UncertainTensor | None
    )


def test_assemble_pipeline_expects_uncertain_values_default_is_None() -> None:
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
def test_assemble_pipeline_actually_runs(n_samples: int) -> None:
    assemble_pipeline(n_samples)


def test_assemble_pipeline_expects_parameter_n_samples() -> None:
    assert "n_samples" in signature(assemble_pipeline).parameters


def test_assemble_pipeline_parameter_n_samples_is_of_type_int() -> None:
    assert signature(assemble_pipeline).parameters["n_samples"].annotation is int


def test_assemble_pipeline_parameter_n_samples_default_is_one() -> None:
    assert signature(assemble_pipeline).parameters["n_samples"].default == 1


def test_assemble_pipeline_states_to_return_anything() -> None:
    assert signature(assemble_pipeline).return_annotation is Any


@pytest.mark.webtest
def test_assemble_pipeline_actually_returns_profiler() -> None:
    assert isinstance(assemble_pipeline(), profile)
