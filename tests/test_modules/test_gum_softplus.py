"""Test the class GUMSoftplus"""
from inspect import signature

import pytest
import torch
from hypothesis import given, HealthCheck, settings, strategies as hst
from numpy.testing import assert_equal
from torch import sigmoid, Tensor
from torch.autograd.profiler import profile
from torch.nn import Module
from torch.nn.functional import softplus
from torch.testing import assert_close  # type: ignore[attr-defined]

from pytorch_gum_uncertainty_propagation import modules
from pytorch_gum_uncertainty_propagation.modules import (
    GUMSoftplus,
)
from pytorch_gum_uncertainty_propagation.uncertainties import (
    _is_positive_semi_definite,
    _is_symmetric,
    UncertainTensor,
)
from .conftest import betas_or_thresholds
from ..conftest import tensors, uncertain_tensors


@pytest.fixture
def gum_softplus_instance() -> GUMSoftplus:
    return GUMSoftplus()


def test_modules_actually_contains_gum_softplus() -> None:
    assert hasattr(modules, "GUMSoftplus")


def test_modules_all_contains_gum_softplus() -> None:
    assert GUMSoftplus.__name__ in modules.__all__


def test_gum_softplus_is_subclass_of_nn_module() -> None:
    assert issubclass(GUMSoftplus, Module)


def test_gum_softplus_has_docstring() -> None:
    assert GUMSoftplus.__doc__ is not None


def test_init_gum_softplus() -> None:
    assert GUMSoftplus()


def test_gum_softplus_has_parameter_beta() -> None:
    assert "beta" in signature(GUMSoftplus).parameters


def test_gum_softplus_parameter_beta_is_annotated_as_float() -> None:
    assert signature(GUMSoftplus).parameters["beta"].annotation is int


def test_gum_softplus_parameter_beta_default_is_one() -> None:
    assert signature(GUMSoftplus).parameters["beta"].default == 1


def test_gum_softplus_has_parameter_threshold() -> None:
    assert "threshold" in signature(GUMSoftplus).parameters


def test_gum_softplus_parameter_threshold_is_annotated_as_float() -> None:
    assert signature(GUMSoftplus).parameters["threshold"].annotation is int


def test_gum_softplus_parameter_threshold_default_is_one() -> None:
    assert signature(GUMSoftplus).parameters["threshold"].default == 20


def test_init_gum_softplus_creates_attribute_softplus(
    gum_softplus_instance: GUMSoftplus,
) -> None:
    assert hasattr(gum_softplus_instance, "_softplus")


def test_init_gum_softplus_creates_attribute_beta(
    gum_softplus_instance: GUMSoftplus,
) -> None:
    assert hasattr(gum_softplus_instance, "beta")


def test_init_gum_softplus_creates_attribute_threshold(
    gum_softplus_instance: GUMSoftplus,
) -> None:
    assert hasattr(gum_softplus_instance, "threshold")


def test_uncertain_quadlu_beta_has_docstring() -> None:
    assert GUMSoftplus.beta.__doc__ is not None


def test_uncertain_quadlu_threshold_has_docstring() -> None:
    assert GUMSoftplus.threshold.__doc__ is not None


@given(hst.integers(min_value=1, max_value=10))
def test_init_gum_softplus_creates_attribute_beta_correctly(beta: int) -> None:
    assert_equal(GUMSoftplus(beta).beta, beta)


@given(hst.integers(min_value=1, max_value=10))
def test_init_gum_softplus_creates_attribute_threshold_correctly(
    threshold: int,
) -> None:
    assert_equal(GUMSoftplus(threshold=threshold).threshold, threshold)


def test_init_gum_softplus_creates_default_attribute_beta_1(
    gum_softplus_instance: GUMSoftplus,
) -> None:
    assert_equal(gum_softplus_instance.beta, 1)


def test_init_gum_softplus_creates_default_attribute_threshold_20(
    gum_softplus_instance: GUMSoftplus,
) -> None:
    assert_equal(gum_softplus_instance.threshold, 20)


def test_gum_softplus_contains_callable_forward() -> None:
    assert callable(GUMSoftplus.forward)


def test_gum_softplus_forward_accepts_one_parameters() -> None:
    assert_equal(len(signature(GUMSoftplus.forward).parameters), 2)


def test_gum_softplus_forward_accepts_uncertain_values_parameter() -> None:
    assert "uncertain_values" in signature(GUMSoftplus.forward).parameters


def test_gum_softplus_forward_expects_uncertain_tensor_parameter() -> None:
    assert issubclass(
        signature(GUMSoftplus.forward).parameters["uncertain_values"].annotation,
        UncertainTensor,
    )


@given(tensors(), hst.integers(min_value=1, max_value=10))
def test_gum_softplus_forward_returns_uncertain_tensor_for_no_uncertainties(
    values: Tensor, out_features: int
) -> None:
    assert isinstance(
        GUMSoftplus(len(values), out_features).forward(
            UncertainTensor(values, None),
        ),
        UncertainTensor,
    )


@given(tensors(), hst.integers(min_value=1, max_value=10))
def test_gum_softplus_forward_triggers_profiler_for_no_uncertainties(
    values: Tensor, out_features: int
) -> None:
    with profile() as profiler:  # type: ignore[no-untyped-call]
        GUMSoftplus(len(values), out_features).forward(
            UncertainTensor(values, None),
        )
    assert "GUMSOFTPLUS PASS" in profiler.key_averages(group_by_stack_n=2).table(
        sort_by="self_cpu_time_total", row_limit=7
    )


@given(uncertain_tensors(), hst.integers(min_value=1, max_value=10))
def test_gum_softplus_forward_returns_uncertain_tensor_for_non_none_uncertainties(
    values_and_uncertainties: UncertainTensor, out_features: int
) -> None:
    assert isinstance(
        GUMSoftplus(len(values_and_uncertainties.values), out_features).forward(
            values_and_uncertainties,
        ),
        UncertainTensor,
    )


@given(tensors(), hst.integers(min_value=1, max_value=10))
def test_gum_softplus_forward_triggers_profiler(
    values: Tensor, out_features: int
) -> None:
    with profile() as profiler:  # type: ignore[no-untyped-call]
        GUMSoftplus(len(values), out_features).forward(
            UncertainTensor(values, None),
        )
    assert "GUMSOFTPLUS PASS" in profiler.key_averages(group_by_stack_n=2).table(
        sort_by="self_cpu_time_total", row_limit=7
    )


@given(tensors())
def test_default_gum_softplus_forward_provides_no_uncertainties_if_not_provided(
    values: Tensor,
) -> None:
    assert GUMSoftplus().forward(UncertainTensor(values, None)).uncertainties is None


@given(uncertain_tensors())
def test_default_gum_softplus_forward_uncertainties_for_random_input(
    uncertain_tensor: UncertainTensor,
) -> None:
    result_uncertainties = GUMSoftplus().forward(uncertain_tensor).uncertainties
    assert result_uncertainties is not None
    first_derivs = torch.sigmoid(uncertain_tensor.values)
    assert_close(
        result_uncertainties,
        first_derivs * uncertain_tensor.uncertainties * first_derivs.unsqueeze(1),
    )


@given(uncertain_tensors(), betas_or_thresholds())
def test_gum_softplus_forward_uncertainties_for_random_input(
    uncertain_tensor: UncertainTensor, beta: int
) -> None:
    result_uncertainties = GUMSoftplus(beta).forward(uncertain_tensor).uncertainties
    assert result_uncertainties is not None
    first_derivs = torch.sigmoid(beta * uncertain_tensor.values)
    assert_close(
        result_uncertainties,
        first_derivs * uncertain_tensor.uncertainties * first_derivs.unsqueeze(1),
    )


@given(uncertain_tensors())
def test_default_gum_softplus_forward_values_for_random_input(
    uncertain_tensor: UncertainTensor,
) -> None:
    gum_softplus = GUMSoftplus()
    assert_close(
        gum_softplus(uncertain_tensor).values,
        (gum_softplus._softplus(uncertain_tensor.values)),
    )


@given(uncertain_tensors(), betas_or_thresholds())
def test_gum_softplus_forward_values_for_random_input(
    uncertain_tensor: UncertainTensor, beta: int
) -> None:
    gum_softplus = GUMSoftplus(beta)
    assert_close(
        gum_softplus(uncertain_tensor).values,
        (gum_softplus._softplus(uncertain_tensor.values)),
    )


@given(uncertain_tensors())
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_default_gum_softplus_forward_for_random_input_is_correct(
    gum_softplus_instance: GUMSoftplus,
    uncertain_tensor: UncertainTensor,
) -> None:
    assert uncertain_tensor.uncertainties is not None
    first_derivs = sigmoid(gum_softplus_instance.beta * uncertain_tensor.values)
    assert_close(
        gum_softplus_instance(uncertain_tensor),
        (
            softplus(uncertain_tensor.values),
            first_derivs * uncertain_tensor.uncertainties * first_derivs.unsqueeze(1),
        ),
    )


@given(uncertain_tensors(), betas_or_thresholds())
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_gum_softplus_forward_for_random_input_is_correct(
    uncertain_tensor: UncertainTensor, beta: int
) -> None:
    gum_softplus_instance = GUMSoftplus(beta)
    assert uncertain_tensor.uncertainties is not None
    first_derivs = sigmoid(beta * uncertain_tensor.values)
    assert_close(
        gum_softplus_instance(uncertain_tensor),
        (
            softplus(uncertain_tensor.values, beta=beta),
            first_derivs * uncertain_tensor.uncertainties * first_derivs.unsqueeze(1),
        ),
    )


@given(uncertain_tensors())
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_default_gum_softplus_forward_results_in_positive_semi_definite_uncertainties(
    gum_softplus_instance: GUMSoftplus,
    uncertain_tensor: UncertainTensor,
) -> None:
    result_uncertainties = gum_softplus_instance.forward(uncertain_tensor).uncertainties
    assert result_uncertainties is not None
    assert _is_positive_semi_definite(result_uncertainties)


@given(uncertain_tensors(), betas_or_thresholds())
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_gum_softplus_forward_results_in_positive_semi_definite_uncertainties(
    uncertain_tensor: UncertainTensor, beta: int
) -> None:
    gum_softplus_instance = GUMSoftplus(beta)
    result_uncertainties = gum_softplus_instance.forward(uncertain_tensor).uncertainties
    assert result_uncertainties is not None
    assert _is_positive_semi_definite(result_uncertainties)


@given(uncertain_tensors())
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_default_gum_softplus_forward_results_in_symmetric_uncertainties(
    gum_softplus_instance: GUMSoftplus,
    uncertain_tensor: UncertainTensor,
) -> None:
    result_uncertainties = gum_softplus_instance.forward(uncertain_tensor).uncertainties
    assert result_uncertainties is not None
    assert _is_symmetric(result_uncertainties)


@given(uncertain_tensors(), betas_or_thresholds())
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_gum_softplus_forward_results_in_symmetric_uncertainties(
    uncertain_tensor: UncertainTensor, beta: int
) -> None:
    gum_softplus_instance = GUMSoftplus(beta)
    result_uncertainties = gum_softplus_instance.forward(uncertain_tensor).uncertainties
    assert result_uncertainties is not None
    assert _is_symmetric(result_uncertainties)
