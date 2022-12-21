from inspect import signature
from typing import Any, Type

import pytest
from hypothesis import given, settings, strategies as hst
from torch import Tensor
from torch.autograd.profiler import profile
from torch.nn import Module

from pytorch_gum_uncertainty_propagation.examples import propagate
from pytorch_gum_uncertainty_propagation.examples.propagate import (
    _construct_partition,
    assemble_pipeline,
)
from pytorch_gum_uncertainty_propagation.modules import (
    GUMQuadLUMLP,
    GUMSigmoidMLP,
    GUMSoftplusMLP,
    QuadLUMLP,
)
from pytorch_gum_uncertainty_propagation.uncertainties import UncertainTensor


def test_propagate_has_docstring() -> None:
    assert propagate.__doc__ is not None


def test_propagate_has_function_assemble_pipeline() -> None:
    assert hasattr(propagate, "assemble_pipeline")


def test_propagate_all_contains_assemble_pipeline() -> None:
    assert assemble_pipeline.__name__ in propagate.__all__


def test_assemble_pipeline_has_docstring() -> None:
    assert assemble_pipeline.__doc__ is not None


def test_assemble_pipeline_expects_parameter_uncertain_values() -> None:
    assert "input_values" in signature(assemble_pipeline).parameters


def test_assemble_pipeline_expects_uncertain_values_of_type_uncertain_values() -> None:
    assert (
        signature(assemble_pipeline).parameters["input_values"].annotation
        == UncertainTensor | Tensor | None
    )


def test_assemble_pipeline_expects_uncertain_values_default_is_None() -> None:
    assert signature(assemble_pipeline).parameters["input_values"].default is None


def test_assemble_pipeline_expects_parameter_mlp_module() -> None:
    assert "generic_mlp_module" in signature(assemble_pipeline).parameters


def test_assemble_pipeline_expects_mlp_module_of_type_uncertain_values() -> None:
    assert (
        signature(assemble_pipeline).parameters["generic_mlp_module"].annotation
        == Type[Module]
    )


def test_construct_partition_exists() -> None:
    assert hasattr(propagate, "_construct_partition")


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
@given(
    hst.sampled_from((GUMQuadLUMLP, GUMSoftplusMLP, GUMSigmoidMLP)),
    hst.integers(min_value=1, max_value=10),
)
@settings(deadline=None)
def test_assemble_pipeline_actually_runs_for_gum_modules(
    mlp_module: Type[Module], n_samples: int
) -> None:
    assemble_pipeline(mlp_module, n_samples)


@pytest.mark.webtest
@given(
    hst.just(QuadLUMLP),
    hst.integers(min_value=1, max_value=10),
)
@settings(deadline=None)
def test_assemble_pipeline_actually_runs_for_quadlu_mlp(
    mlp_module: Type[Module], n_samples: int
) -> None:
    assemble_pipeline(mlp_module, n_samples)


def test_assemble_pipeline_expects_parameter_n_samples() -> None:
    assert "n_samples" in signature(assemble_pipeline).parameters


def test_assemble_pipeline_parameter_n_samples_is_of_type_int() -> None:
    assert signature(assemble_pipeline).parameters["n_samples"].annotation is int


def test_assemble_pipeline_parameter_n_samples_default_is_one() -> None:
    assert signature(assemble_pipeline).parameters["n_samples"].default == 1


def test_assemble_pipeline_states_to_return_anything() -> None:
    assert signature(assemble_pipeline).return_annotation is Any


@pytest.mark.webtest
@given(hst.sampled_from((GUMQuadLUMLP, GUMSoftplusMLP, GUMSigmoidMLP)))
@settings(deadline=None)
def test_assemble_pipeline_actually_returns_profiler(
    activation_module: Type[Module],
) -> None:
    assert isinstance(assemble_pipeline(activation_module), profile)
