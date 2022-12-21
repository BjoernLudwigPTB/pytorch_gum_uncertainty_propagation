"""Test the class GUMSoftplusMLP"""
from inspect import signature
from itertools import islice
from typing import cast

import torch
from hypothesis import given, strategies as hst
from hypothesis.strategies import composite, DrawFn, SearchStrategy
from numpy.testing import assert_equal
from torch.nn import ModuleList, Softplus
from torch.testing import assert_close  # type: ignore[attr-defined]

from pytorch_gum_uncertainty_propagation import modules
from pytorch_gum_uncertainty_propagation.modules import (
    GUMSoftplus,
    GUMSoftplusMLP,
    MLP,
    UncertainLinear,
)
from pytorch_gum_uncertainty_propagation.uncertainties import (
    UncertainTensor,
)
from .conftest import betas_or_thresholds
from ..conftest import uncertain_tensors


@composite
def gum_softplus_mlps(
    draw: DrawFn,
    in_dimen: int | None = None,
    n_hidden_channels: int | None = None,
    out_channels: int | None = None,
) -> SearchStrategy[GUMSoftplusMLP]:
    dimen_strategy = hst.integers(min_value=1, max_value=100)
    if in_dimen is None:
        in_dimen = draw(dimen_strategy)
    if n_hidden_channels is None:
        n_hidden_channels = draw(dimen_strategy)
    hidden_and_out_dimens = []
    for _ in range(n_hidden_channels - 1):
        hidden_and_out_dimens.append(draw(dimen_strategy))
    if out_channels is None:
        hidden_and_out_dimens.append(draw(dimen_strategy))
    else:
        hidden_and_out_dimens.append(out_channels)
    return cast(
        SearchStrategy[GUMSoftplusMLP],
        GUMSoftplusMLP(in_dimen, hidden_and_out_dimens),
    )


def test_modules_contains_gum_softplus_mlp() -> None:
    assert hasattr(modules, "GUMSoftplusMLP")


def test_modules_all_contains_gum_softplus_mlp() -> None:
    assert GUMSoftplusMLP.__name__ in modules.__all__


def test_gum_softplus_mlp_is_subclass_of_nn_module() -> None:
    assert issubclass(GUMSoftplusMLP, MLP)


def test_gum_softplus_mlp_has_docstring() -> None:
    assert GUMSoftplusMLP.__doc__ is not None


def test_gum_softplus_mlp_expects_in_channels_parameter() -> None:
    assert "in_features" in signature(GUMSoftplusMLP).parameters


def test_gum_softplus_mlp_expects_in_features_parameter_as_int() -> None:
    assert issubclass(
        signature(GUMSoftplusMLP).parameters["in_features"].annotation, int
    )


def test_gum_softplus_mlp_expects_out_features_parameter() -> None:
    assert "out_features" in signature(GUMSoftplusMLP).parameters


def test_gum_softplus_mlp_expects_out_features_parameter_as_int_list() -> None:
    assert signature(GUMSoftplusMLP).parameters["out_features"].annotation == list[int]


def test_gum_softplus_mlp_expects_beta_parameter() -> None:
    assert "beta" in signature(GUMSoftplusMLP).parameters


def test_gum_softplus_mlp_expects_beta_parameter_as_int() -> None:
    assert signature(GUMSoftplusMLP).parameters["beta"].annotation == int


def test_gum_softplus_mlp_default_for_beta_parameter_is_one() -> None:
    assert signature(GUMSoftplusMLP).parameters["beta"].default == 1


def test_gum_softplus_mlp_expects_threshold_parameter() -> None:
    assert "threshold" in signature(GUMSoftplusMLP).parameters


def test_gum_softplus_mlp_expects_threshold_parameter_as_int() -> None:
    assert signature(GUMSoftplusMLP).parameters["threshold"].annotation == int


def test_gum_softplus_mlp_default_for_threshold_parameter_is_one() -> None:
    assert signature(GUMSoftplusMLP).parameters["threshold"].default == 20


@given(gum_softplus_mlps())
def test_gum_softplus_mlp_is_gum_softplus_mlp(
    gum_softplus_mlp: GUMSoftplusMLP,
) -> None:
    assert isinstance(gum_softplus_mlp, GUMSoftplusMLP)


@given(gum_softplus_mlps())
def test_gum_softplus_mlp_children_does_not_provide_access_to_module_list(
    gum_softplus_mlp: GUMSoftplusMLP,
) -> None:
    assert type(next(gum_softplus_mlp.children())) is not ModuleList


@given(gum_softplus_mlps())
def test_gum_softplus_mlp_children_provides_access_to_uncertain_linears_and_quadlus(
    gum_softplus_mlp: GUMSoftplusMLP,
) -> None:
    for child in gum_softplus_mlp.children():
        assert type(child) is UncertainLinear or type(child) is GUMSoftplus


def test_gum_softplus_mlp_children_has_docstring() -> None:
    assert GUMSoftplusMLP.children.__doc__ is not None


@given(gum_softplus_mlps(in_dimen=5))
def test_init_gum_softplus_mlp_input_layer_as_specified(
    gum_softplus_mlp: GUMSoftplusMLP,
) -> None:
    assert_equal(next(gum_softplus_mlp.children()).in_features, 5)


@given(gum_softplus_mlps(n_hidden_channels=3))
def test_init_gum_softplus_mlp_input_dimension_as_specified(
    gum_softplus_mlp: GUMSoftplusMLP,
) -> None:
    assert_equal(len(gum_softplus_mlp), 2 * 3)


@given(gum_softplus_mlps())
def test_init_gum_softplus_mlp_each_pair_of_layers_is_linear_and_uncertain_quadlu(
    gum_softplus_mlp: GUMSoftplusMLP,
) -> None:
    layer_iter = gum_softplus_mlp.children()
    while first_layer_in_a_pair := next(layer_iter, None):
        assert isinstance(first_layer_in_a_pair, UncertainLinear)
        second_layer_in_a_pair = next(layer_iter)
        assert isinstance(second_layer_in_a_pair, GUMSoftplus)


def test_gum_softplus_mlp_has_attribute_forward() -> None:
    assert hasattr(GUMSoftplusMLP, "forward")


def test_gum_softplus_mlp_forward_expects_two_parameters() -> None:
    assert_equal(len(signature(GUMSoftplusMLP.forward).parameters), 2)


@given(uncertain_tensors(length=8), gum_softplus_mlps(in_dimen=8))
def test_gum_softplus_mlp_outputs_tuple_tensor(
    values_and_uncertainties: UncertainTensor,
    gum_softplus_mlp: GUMSoftplusMLP,
) -> None:
    assert isinstance(gum_softplus_mlp(values_and_uncertainties), UncertainTensor)


@given(
    hst.lists(hst.integers(min_value=1, max_value=10), min_size=1, max_size=10),
    uncertain_tensors(),
)
def test_gum_softplus_mlp_correct_output_dimension_of_values(
    out_dimens: list[int],
    uncertain_values: UncertainTensor,
) -> None:
    gum_softplus_mlp = GUMSoftplusMLP(len(uncertain_values.values), out_dimens)
    assert_equal(
        len(gum_softplus_mlp(uncertain_values)[0]),
        out_dimens[-1],
    )


@given(
    hst.lists(hst.integers(min_value=1, max_value=10), min_size=1, max_size=10),
    uncertain_tensors(),
)
def test_gum_softplus_mlp_correct_output_dimension_of_uncertainties(
    out_dimens: list[int],
    uncertain_values: UncertainTensor,
) -> None:
    gum_softplus_mlp = GUMSoftplusMLP(len(uncertain_values.values), out_dimens)
    assert_equal(
        len(gum_softplus_mlp(uncertain_values)[1]),
        out_dimens[-1],
    )


@given(hst.lists(hst.integers(min_value=1, max_value=10), min_size=1, max_size=10))
def test_gum_softplus_mlp_correct_output_dimension_of_all_layers(
    out_dimens: list[int],
) -> None:
    gum_softplus_mlp = GUMSoftplusMLP(4, out_dimens)
    linear_layers_iter = islice(gum_softplus_mlp.children(), None, None, 2)
    out_dimens_iter = iter(out_dimens)
    for linear_layer in linear_layers_iter:
        assert_equal(
            linear_layer.out_features,
            next(out_dimens_iter),
        )


@given(
    uncertain_tensors(length=9),
    gum_softplus_mlps(in_dimen=9, n_hidden_channels=1, out_channels=9),
)
def test_gum_softplus_mlp_is_correct_for_identity_matrix_product(
    uncertain_values: UncertainTensor,
    gum_softplus_mlp: GUMSoftplusMLP,
) -> None:
    gum_softplus_mlp[0].weight.data.zero_()
    gum_softplus_mlp[0].weight.data.fill_diagonal_(1.0)
    assert_close(
        gum_softplus_mlp(UncertainTensor(uncertain_values.values, None)),
        UncertainTensor(
            Softplus().forward(
                uncertain_values.values @ torch.eye(9) + gum_softplus_mlp[0].bias
            ),
            None,
        ),
    )


@given(
    hst.integers(min_value=1, max_value=10),
    hst.lists(hst.integers(min_value=1, max_value=10), min_size=1, max_size=10),
    betas_or_thresholds(),
)
def test_gum_softplus_mlp_can_be_instantiated_with_keyword_argument_beta(
    in_features: int, out_features: list[int], beta: int
) -> None:
    assert_equal(MLP(in_features, out_features, GUMSoftplus, beta=beta)[1].beta, beta)


@given(
    hst.integers(min_value=1, max_value=10),
    hst.lists(hst.integers(min_value=1, max_value=10), min_size=1, max_size=10),
    betas_or_thresholds(),
)
def test_gum_softplus_mlp_can_be_instantiated_with_keyword_argument_threshold(
    in_features: int, out_features: list[int], threshold: int
) -> None:
    assert_equal(
        GUMSoftplusMLP(in_features, out_features, threshold=threshold)[1].threshold,
        threshold,
    )


@given(
    hst.integers(min_value=1, max_value=10),
    hst.lists(hst.integers(min_value=1, max_value=10), min_size=1, max_size=10),
    betas_or_thresholds(),
    betas_or_thresholds(),
)
def test_gum_softplus_mlp_init_with_positional_argument_for_threshold(
    in_features: int, out_features: list[int], beta: int, threshold: int
) -> None:
    gum_softplus_mlp = GUMSoftplusMLP(in_features, out_features, beta, threshold)
    assert_equal(gum_softplus_mlp[1].threshold, threshold)
