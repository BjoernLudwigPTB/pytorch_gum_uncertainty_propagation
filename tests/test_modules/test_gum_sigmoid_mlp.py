"""Test the class GUMSigmoidMLP"""
from inspect import signature
from itertools import islice
from typing import cast

import torch
from hypothesis import given, strategies as hst
from hypothesis.strategies import composite, DrawFn, SearchStrategy
from numpy.testing import assert_equal
from torch import sigmoid
from torch.nn import ModuleList
from torch.testing import assert_close  # type: ignore[attr-defined]

from pytorch_gum_uncertainty_propagation import modules
from pytorch_gum_uncertainty_propagation.modules import (
    GUMSigmoid,
    GUMSigmoidMLP,
    MLP,
    GUMLinear,
)
from pytorch_gum_uncertainty_propagation.uncertainties import (
    UncertainTensor,
)
from ..conftest import uncertain_tensors


@composite
def gum_sigmoid_mlps(
    draw: DrawFn,
    in_dimen: int | None = None,
    n_hidden_channels: int | None = None,
    out_channels: int | None = None,
) -> SearchStrategy[GUMSigmoidMLP]:
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
        SearchStrategy[GUMSigmoidMLP],
        GUMSigmoidMLP(in_dimen, hidden_and_out_dimens),
    )


def test_modules_contains_gum_sigmoid_mlp() -> None:
    assert hasattr(modules, "GUMSigmoidMLP")


def test_modules_all_contains_gum_sigmoid_mlp() -> None:
    assert GUMSigmoidMLP.__name__ in modules.__all__


def test_gum_sigmoid_mlp_is_subclass_of_nn_module() -> None:
    assert issubclass(GUMSigmoidMLP, MLP)


def test_gum_sigmoid_mlp_has_docstring() -> None:
    assert GUMSigmoidMLP.__doc__ is not None


def test_gum_sigmoid_mlp_expects_in_features_parameter() -> None:
    assert "in_features" in signature(GUMSigmoidMLP).parameters


def test_gum_sigmoid_mlp_expects_in_features_parameter_as_int() -> None:
    assert issubclass(
        signature(GUMSigmoidMLP).parameters["in_features"].annotation, int
    )


def test_gum_sigmoid_mlp_expects_out_features_parameter() -> None:
    assert "out_features" in signature(GUMSigmoidMLP).parameters


def test_gum_sigmoid_mlp_expects_out_features_parameter_as_int_list() -> None:
    assert signature(GUMSigmoidMLP).parameters["out_features"].annotation == list[int]


@given(gum_sigmoid_mlps())
def test_gum_sigmoid_mlp_is_gum_sigmoid_mlp(
    gum_sigmoid_mlp: GUMSigmoidMLP,
) -> None:
    assert isinstance(gum_sigmoid_mlp, GUMSigmoidMLP)


def test_gum_sigmoid_mlp_init_has_docstring() -> None:
    assert GUMSigmoidMLP.__init__.__doc__ is not None


@given(gum_sigmoid_mlps())
def test_gum_sigmoid_mlp_children_does_not_provide_access_to_module_list(
    gum_sigmoid_mlp: GUMSigmoidMLP,
) -> None:
    assert type(next(gum_sigmoid_mlp.children())) is not ModuleList


@given(gum_sigmoid_mlps())
def test_gum_sigmoid_mlp_children_provides_access_to_uncertain_linears_and_quadlus(
    gum_sigmoid_mlp: GUMSigmoidMLP,
) -> None:
    for child in gum_sigmoid_mlp.children():
        assert type(child) is GUMLinear or type(child) is GUMSigmoid


@given(gum_sigmoid_mlps(in_dimen=5))
def test_init_gum_sigmoid_mlp_input_layer_as_specified(
    gum_sigmoid_mlp: GUMSigmoidMLP,
) -> None:
    assert_equal(next(gum_sigmoid_mlp.children()).in_features, 5)


@given(gum_sigmoid_mlps(n_hidden_channels=3))
def test_init_gum_sigmoid_mlp_input_dimension_as_specified(
    gum_sigmoid_mlp: GUMSigmoidMLP,
) -> None:
    assert_equal(len(gum_sigmoid_mlp), 2 * 3)


@given(gum_sigmoid_mlps())
def test_init_gum_sigmoid_mlp_each_pair_of_layers_is_linear_and_gum_sigmoid(
    gum_sigmoid_mlp: GUMSigmoidMLP,
) -> None:
    layer_iter = gum_sigmoid_mlp.children()
    while first_layer_in_a_pair := next(layer_iter, None):
        assert isinstance(first_layer_in_a_pair, GUMLinear)
        second_layer_in_a_pair = next(layer_iter)
        assert isinstance(second_layer_in_a_pair, GUMSigmoid)


def test_gum_sigmoid_mlp_has_attribute_forward() -> None:
    assert hasattr(GUMSigmoidMLP, "forward")


def test_gum_sigmoid_mlp_forward_expects_two_parameters() -> None:
    assert_equal(len(signature(GUMSigmoidMLP.forward).parameters), 2)


@given(uncertain_tensors(length=8), gum_sigmoid_mlps(in_dimen=8))
def test_gum_sigmoid_mlp_outputs_uncertain_tensor(
    values_and_uncertainties: UncertainTensor,
    gum_sigmoid_mlp: GUMSigmoidMLP,
) -> None:
    assert isinstance(gum_sigmoid_mlp(values_and_uncertainties), UncertainTensor)


@given(
    hst.lists(hst.integers(min_value=1, max_value=10), min_size=1, max_size=10),
    uncertain_tensors(),
)
def test_gum_sigmoid_mlp_correct_output_dimension_of_values(
    out_dimens: list[int],
    uncertain_values: UncertainTensor,
) -> None:
    gum_sigmoid_mlp = GUMSigmoidMLP(len(uncertain_values.values), out_dimens)
    assert_equal(
        len(gum_sigmoid_mlp(uncertain_values)[0]),
        out_dimens[-1],
    )


@given(
    hst.lists(hst.integers(min_value=1, max_value=10), min_size=1, max_size=10),
    uncertain_tensors(),
)
def test_gum_sigmoid_mlp_correct_output_dimension_of_uncertainties(
    out_dimens: list[int],
    uncertain_values: UncertainTensor,
) -> None:
    gum_sigmoid_mlp = GUMSigmoidMLP(len(uncertain_values.values), out_dimens)
    assert_equal(
        len(gum_sigmoid_mlp(uncertain_values)[1]),
        out_dimens[-1],
    )


@given(hst.lists(hst.integers(min_value=1, max_value=10), min_size=1, max_size=10))
def test_gum_sigmoid_mlp_correct_output_dimension_of_all_layers(
    out_dimens: list[int],
) -> None:
    gum_sigmoid_mlp = GUMSigmoidMLP(4, out_dimens)
    linear_layers_iter = islice(gum_sigmoid_mlp.children(), None, None, 2)
    out_dimens_iter = iter(out_dimens)
    for linear_layer in linear_layers_iter:
        assert_equal(
            linear_layer.out_features,
            next(out_dimens_iter),
        )


@given(
    uncertain_tensors(length=9),
    gum_sigmoid_mlps(in_dimen=9, n_hidden_channels=1, out_channels=9),
)
def test_gum_sigmoid_mlp_is_correct_for_identity_matrix_product(
    uncertain_values: UncertainTensor,
    gum_sigmoid_mlp: GUMSigmoidMLP,
) -> None:
    gum_sigmoid_mlp[0].weight.data.zero_()
    gum_sigmoid_mlp[0].weight.data.fill_diagonal_(1.0)
    assert_close(
        gum_sigmoid_mlp(UncertainTensor(uncertain_values.values, None)),
        UncertainTensor(
            sigmoid(uncertain_values.values @ torch.eye(9) + gum_sigmoid_mlp[0].bias),
            None,
        ),
    )
