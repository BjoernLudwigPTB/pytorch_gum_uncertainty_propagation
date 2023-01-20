import os
from glob import glob
from inspect import signature
from itertools import chain
from typing import Any, Callable, Generator, Type

import pytest
from hypothesis import given, settings, strategies as hst
from torch.autograd.profiler import profile
from torch.nn import Module

from pytorch_gum_uncertainty_propagation.examples import propagate
from pytorch_gum_uncertainty_propagation.examples.propagate import (
    _construct_out_features_counts,
    assemble_pipeline,
    iterate_over_activations_and_architectures,
)
from pytorch_gum_uncertainty_propagation.modules import (
    GUMQuadLUMLP,
    GUMSigmoidMLP,
    GUMSoftplusMLP,
)


@pytest.fixture(scope="module")
def file_deleter() -> Callable[[tuple[str, ...]], None]:
    def deleter(endings: tuple[str, ...]) -> None:
        for file in chain(*(glob(f"*{ending}") for ending in endings)):
            try:
                os.remove(file)
            except FileNotFoundError:
                pass

    return deleter


@pytest.fixture
def cleanup_traces_after_run(
    file_deleter: Callable[[tuple[str, ...]], None]
) -> Generator[None, None, None]:
    yield
    file_deleter(("*_layers_trace.json",))


def test_propagate_has_docstring() -> None:
    assert propagate.__doc__ is not None


def test_propagate_has_function_assemble_pipeline() -> None:
    assert hasattr(propagate, "assemble_pipeline")


def test_propagate_all_contains_assemble_pipeline() -> None:
    assert assemble_pipeline.__name__ in propagate.__all__


def test_assemble_pipeline_has_docstring() -> None:
    assert assemble_pipeline.__doc__ is not None


def test_assemble_pipeline_expects_parameter_mlp_module() -> None:
    assert "generic_mlp_module" in signature(assemble_pipeline).parameters


def test_assemble_pipeline_expects_mlp_module_of_type_uncertain_values() -> None:
    assert (
        signature(assemble_pipeline).parameters["generic_mlp_module"].annotation
        == Type[Module]
    )


def test_construct_out_features_counts_exists() -> None:
    assert hasattr(propagate, "_construct_out_features_counts")


def test_construct_out_features_counts_has_docstring() -> None:
    assert _construct_out_features_counts.__doc__ is not None


def test_construct_out_features_counts_has_parameter_in_features() -> None:
    assert "in_features" in signature(_construct_out_features_counts).parameters


def test_construct_out_features_counts_has_parameter_out_features() -> None:
    assert "out_features" in signature(_construct_out_features_counts).parameters


def test_construct_out_features_counts_parameter_out_features_is_of_type_int() -> None:
    assert (
        signature(_construct_out_features_counts).parameters["out_features"].annotation
        is int
    )


def test_construct_out_features_counts_has_parameter_depth() -> None:
    assert "depth" in signature(_construct_out_features_counts).parameters


def test_construct_out_features_counts_parameter_depth_is_of_type_int_or_none() -> None:
    assert (
        signature(_construct_out_features_counts).parameters["depth"].annotation is int
    )


def test_construct_out_features_counts_parameter_in_features_is_of_type_int() -> None:
    assert (
        signature(_construct_out_features_counts).parameters["in_features"].annotation
        is int
    )


def test_construct_out_features_counts_parameter_states_to_return_int_list() -> None:
    assert (
        signature(_construct_out_features_counts).return_annotation == tuple[int, ...]
    )


@given(hst.integers(min_value=1, max_value=10))
def test_construct_out_features_counts_actually_returns_int_list(
    in_features: int,
) -> None:
    assert isinstance(_construct_out_features_counts(in_features), tuple)


@given(hst.integers(min_value=1, max_value=100))
def test_construct_out_features_counts_returns_non_empty_list(in_features: int) -> None:
    assert len(_construct_out_features_counts(in_features))


def test_construct_out_features_counts_returns_correct_small_example() -> None:
    assert _construct_out_features_counts(89, 8, 6) == (76, 63, 50, 36, 22, 8)


@given(hst.integers(min_value=1, max_value=100))
def test_construct_out_features_counts_is_descending(in_features: int) -> None:
    partition = _construct_out_features_counts(in_features)
    assert partition[:] > partition[1:]


def test_construct_out_features_counts_returns_correct_large_example() -> None:
    assert _construct_out_features_counts(99, 3, 59) == (
        98,
        97,
        96,
        95,
        94,
        93,
        92,
        91,
        90,
        89,
        88,
        87,
        86,
        85,
        84,
        83,
        82,
        81,
        80,
        79,
        78,
        77,
        75,
        73,
        71,
        69,
        67,
        65,
        63,
        61,
        59,
        57,
        55,
        53,
        51,
        49,
        47,
        45,
        43,
        41,
        39,
        37,
        35,
        33,
        31,
        29,
        27,
        25,
        23,
        21,
        19,
        17,
        15,
        13,
        11,
        9,
        7,
        5,
        3,
    )


@pytest.mark.webtest
@given(
    hst.sampled_from((GUMQuadLUMLP, GUMSoftplusMLP, GUMSigmoidMLP)),
    hst.integers(min_value=1, max_value=3),
    hst.integers(min_value=1, max_value=10),
    hst.integers(min_value=1, max_value=3),
    hst.booleans(),
)
@settings(deadline=None)
def test_assemble_pipeline_actually_runs_for_gum_modules(
    mlp_module: Type[Module],
    size_scaler: int,
    idx_start: int,
    depth: int,
    uncertainties_to_none: bool,
) -> None:
    assemble_pipeline(mlp_module, size_scaler, idx_start, depth, uncertainties_to_none)


def test_assemble_pipeline_expects_parameter_size_scaler() -> None:
    assert "size_scaler" in signature(assemble_pipeline).parameters


def test_assemble_pipeline_parameter_size_scaler_is_of_type_int() -> None:
    assert signature(assemble_pipeline).parameters["size_scaler"].annotation is int


def test_assemble_pipeline_parameter_size_scaler_default_is_one() -> None:
    assert signature(assemble_pipeline).parameters["size_scaler"].default == 1


def test_assemble_pipeline_expects_parameter_idx_start() -> None:
    assert "idx_start" in signature(assemble_pipeline).parameters


def test_assemble_pipeline_parameter_idx_start_is_of_type_int() -> None:
    assert signature(assemble_pipeline).parameters["idx_start"].annotation is int


def test_assemble_pipeline_parameter_idx_start_default_is_zero() -> None:
    assert signature(assemble_pipeline).parameters["idx_start"].default == 0


def test_assemble_pipeline_expects_parameter_set_uncertainties_to_none() -> None:
    assert "set_uncertainties_to_none" in signature(assemble_pipeline).parameters


def test_assemble_pipeline_parameter_set_uncertainties_to_none_is_type_bool() -> None:
    assert (
        signature(assemble_pipeline).parameters["set_uncertainties_to_none"].annotation
        is bool
    )


def test_assemble_pipeline_parameter_set_uncertainties_to_none_default_false() -> None:
    assert (
        signature(assemble_pipeline).parameters["set_uncertainties_to_none"].default
        is False
    )


def test_assemble_pipeline_expects_parameter_depth() -> None:
    assert "depth" in signature(assemble_pipeline).parameters


def test_assemble_pipeline_parameter_depth_is_of_type_int() -> None:
    assert signature(assemble_pipeline).parameters["depth"].annotation is int


def test_assemble_pipeline_parameter_depth_default_is_two() -> None:
    assert signature(assemble_pipeline).parameters["depth"].default == 1


def test_assemble_pipeline_states_to_return_anything() -> None:
    assert signature(assemble_pipeline).return_annotation is Any


@pytest.mark.webtest
@given(hst.sampled_from((GUMQuadLUMLP, GUMSoftplusMLP, GUMSigmoidMLP)))
@settings(deadline=None)
def test_assemble_pipeline_actually_returns_profiler(
    activation_module: Type[Module],
) -> None:
    assert isinstance(assemble_pipeline(activation_module), profile)


def test_iterate_over_activations_and_architectures_runs(
    cleanup_traces_after_run: Generator[None, None, None],
) -> None:
    iterate_over_activations_and_architectures((1, 2), (1, 2, 10))
