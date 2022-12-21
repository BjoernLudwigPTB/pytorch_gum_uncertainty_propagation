"""Test the class GUMSigmoid"""
from inspect import signature

import pytest
import torch
from hypothesis import given, HealthCheck, settings
from numpy.testing import assert_equal
from torch import Tensor
from torch.autograd.profiler import profile
from torch.nn import Module, Sigmoid
from torch.testing import assert_close  # type: ignore[attr-defined]

from pytorch_gum_uncertainty_propagation import modules
from pytorch_gum_uncertainty_propagation.modules import (
    GUMSigmoid,
)
from pytorch_gum_uncertainty_propagation.uncertainties import (
    _is_positive_semi_definite,
    _is_symmetric,
    UncertainTensor,
)
from ..conftest import tensors, uncertain_tensors


@pytest.fixture
def gum_sigmoid_instance() -> GUMSigmoid:
    return GUMSigmoid()


def test_modules_actually_contains_gum_sigmoid() -> None:
    assert hasattr(modules, "GUMSigmoid")


def test_modules_all_contains_gum_sigmoid() -> None:
    assert GUMSigmoid.__name__ in modules.__all__


def test_gum_sigmoid_is_subclass_of_nn_module() -> None:
    assert issubclass(GUMSigmoid, Module)


def test_gum_sigmoid_has_docstring() -> None:
    assert GUMSigmoid.__doc__ is not None


def test_gum_sigmoid_init_has_docstring() -> None:
    assert GUMSigmoid.__init__.__doc__ is not None


def test_init_gum_sigmoid() -> None:
    assert GUMSigmoid()


def test_init_gum_sigmoid_creates_attribute_sigmoid(
    gum_sigmoid_instance: GUMSigmoid,
) -> None:
    assert hasattr(gum_sigmoid_instance, "_sigmoid")


def test_init_gum_sigmoid_creates_attribute_sigmoid_as_sigmoid(
    gum_sigmoid_instance: GUMSigmoid,
) -> None:
    assert isinstance(gum_sigmoid_instance._sigmoid, Sigmoid)


def test_gum_sigmoid_contains_callable_forward() -> None:
    assert callable(GUMSigmoid.forward)


def test_gum_sigmoid_forward_accepts_one_parameters() -> None:
    assert_equal(len(signature(GUMSigmoid.forward).parameters), 2)


def test_gum_sigmoid_forward_accepts_uncertain_values_parameter() -> None:
    assert "uncertain_values" in signature(GUMSigmoid.forward).parameters


def test_gum_sigmoid_forward_expects_uncertain_tensor_parameter() -> None:
    assert issubclass(
        signature(GUMSigmoid.forward).parameters["uncertain_values"].annotation,
        UncertainTensor,
    )


def test_gum_sigmoid_forward_states_to_return_uncertain_tensor() -> None:
    assert signature(GUMSigmoid.forward).return_annotation is UncertainTensor


def test_gum_sigmoid_forward_has_docstring() -> None:
    assert GUMSigmoid.forward.__doc__ is not None


@given(tensors())
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_gum_sigmoid_forward_returns_uncertain_tensor_for_no_uncertainties(
    gum_sigmoid_instance: GUMSigmoid,
    values: Tensor,
) -> None:
    assert isinstance(
        gum_sigmoid_instance.forward(
            UncertainTensor(values, None),
        ),
        UncertainTensor,
    )


@given(tensors())
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_gum_sigmoid_forward_triggers_profiler_for_no_uncertainties(
    gum_sigmoid_instance: GUMSigmoid,
    values: Tensor,
) -> None:
    with profile() as profiler:  # type: ignore[no-untyped-call]
        gum_sigmoid_instance.forward(
            UncertainTensor(values, None),
        )
    assert "GUMSIGMOID PASS" in profiler.key_averages(group_by_stack_n=2).table(
        sort_by="self_cpu_time_total", row_limit=7
    )


@given(uncertain_tensors())
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_gum_sigmoid_forward_returns_uncertain_tensor_for_non_none_uncertainties(
    gum_sigmoid_instance: GUMSigmoid,
    values_and_uncertainties: UncertainTensor,
) -> None:
    assert isinstance(
        gum_sigmoid_instance.forward(
            values_and_uncertainties,
        ),
        UncertainTensor,
    )


@given(tensors())
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_gum_sigmoid_forward_triggers_profiler(
    gum_sigmoid_instance: GUMSigmoid,
    values: Tensor,
) -> None:
    with profile() as profiler:  # type: ignore[no-untyped-call]
        gum_sigmoid_instance.forward(
            UncertainTensor(values, None),
        )
    assert "GUMSIGMOID PASS" in profiler.key_averages(group_by_stack_n=2).table(
        sort_by="self_cpu_time_total", row_limit=7
    )


@given(tensors())
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_gum_sigmoid_forward_provides_no_uncertainties_if_not_provided(
    gum_sigmoid_instance: GUMSigmoid,
    values: Tensor,
) -> None:
    assert (
        gum_sigmoid_instance.forward(UncertainTensor(values, None)).uncertainties
        is None
    )


@given(uncertain_tensors())
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_gum_sigmoid_forward_uncertainties_for_random_input(
    gum_sigmoid_instance: GUMSigmoid, uncertain_tensor: UncertainTensor
) -> None:
    result_uncertainties = gum_sigmoid_instance.forward(uncertain_tensor).uncertainties
    assert result_uncertainties is not None
    sigmoid_of_x = torch.sigmoid(uncertain_tensor.values)
    first_derivs = sigmoid_of_x * (1 - sigmoid_of_x)
    assert_close(
        result_uncertainties,
        first_derivs * uncertain_tensor.uncertainties * first_derivs.unsqueeze(1),
    )


@given(uncertain_tensors())
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_gum_sigmoid_forward_values_for_random_input(
    gum_sigmoid_instance: GUMSigmoid, uncertain_tensor: UncertainTensor
) -> None:
    assert_close(
        gum_sigmoid_instance(uncertain_tensor).values,
        (gum_sigmoid_instance._sigmoid(uncertain_tensor.values)),
    )


@given(uncertain_tensors())
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_gum_sigmoid_forward_for_random_input(
    gum_sigmoid_instance: GUMSigmoid, uncertain_tensor: UncertainTensor
) -> None:
    result = gum_sigmoid_instance.forward(uncertain_tensor)
    assert result.uncertainties is not None
    sigmoid_of_x = torch.sigmoid(uncertain_tensor.values)
    first_derivs = sigmoid_of_x * (1 - sigmoid_of_x)
    assert_close(
        result,
        UncertainTensor(
            torch.sigmoid(uncertain_tensor.values),
            first_derivs * uncertain_tensor.uncertainties * first_derivs.unsqueeze(1),
        ),
    )


@given(uncertain_tensors())
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_gum_sigmoid_forward_results_in_positive_semi_definite_uncertainties(
    gum_sigmoid_instance: GUMSigmoid,
    uncertain_tensor: UncertainTensor,
) -> None:
    result_uncertainties = gum_sigmoid_instance.forward(uncertain_tensor).uncertainties
    assert result_uncertainties is not None
    assert _is_positive_semi_definite(result_uncertainties)


@given(uncertain_tensors())
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_gum_sigmoid_forward_results_in_symmetric_uncertainties(
    gum_sigmoid_instance: GUMSigmoid,
    uncertain_tensor: UncertainTensor,
) -> None:
    result_uncertainties = gum_sigmoid_instance.forward(uncertain_tensor).uncertainties
    assert result_uncertainties is not None
    assert _is_symmetric(result_uncertainties)
