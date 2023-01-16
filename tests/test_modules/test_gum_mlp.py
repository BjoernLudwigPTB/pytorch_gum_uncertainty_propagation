"""Test the class MLP"""
from inspect import signature
from itertools import islice
from typing import cast, Type

import torch
from hypothesis import given, settings, strategies as hst
from hypothesis.strategies import composite, DrawFn, SearchStrategy
from numpy.testing import assert_equal
from torch import Tensor
from torch.nn import Linear, Module, Sequential, Sigmoid, Softplus
from torch.testing import assert_close  # type: ignore[attr-defined]

from pytorch_gum_uncertainty_propagation import modules
from pytorch_gum_uncertainty_propagation.modules import (
    GUMSoftplus,
    MLP,
    QuadLU,
    GUMLinear,
    GUMQuadLU,
)
from pytorch_gum_uncertainty_propagation.uncertainties import (
    UncertainTensor,
)
from .conftest import betas_or_thresholds
from ..conftest import tensors, uncertain_tensors


@composite
def mlps(
    draw: DrawFn,
    in_dimen: int | None = None,
    n_hidden_channels: int | None = None,
    out_channels: int | None = None,
    activation: SearchStrategy[type[Module]] | None = None,
    uncertain_inputs_exclusively: bool = False,
) -> SearchStrategy[MLP]:
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
    if activation is None:
        possible_activations = [GUMQuadLU, GUMSoftplus]
        if not uncertain_inputs_exclusively:
            possible_activations.extend([QuadLU, Softplus, Sigmoid])
        activation_module = draw(hst.sampled_from(possible_activations))
    else:
        activation_module = draw(activation)
    return cast(
        SearchStrategy[MLP],
        MLP(in_dimen, hidden_and_out_dimens, activation_module),
    )


def test_modules_contains_mlp() -> None:
    assert hasattr(modules, "MLP")


def test_modules_all_contains_mlp() -> None:
    assert MLP.__name__ in modules.__all__


def test_gum_mlp_is_subclass_of_nn_sequential() -> None:
    assert issubclass(MLP, Sequential)


def test_gum_mlp_has_docstring() -> None:
    assert MLP.__doc__ is not None


def test_gum_mlp_init_has_docstring() -> None:
    assert MLP.__init__.__doc__ is not None


def test_gum_mlp_expects_in_features_parameter() -> None:
    assert "in_features" in signature(MLP).parameters


def test_gum_mlp_expects_args_parameter() -> None:
    assert "args" in signature(MLP).parameters


def test_gum_mlp_expects_args_parameter_as_dict_of_str_and_anys() -> None:
    assert signature(MLP).parameters["args"].annotation == int | float


def test_gum_mlp_expects_kwargs_parameter() -> None:
    assert "kwargs" in signature(MLP).parameters


def test_gum_mlp_expects_kwargs_parameter_as_dict_of_str_and_anys() -> None:
    assert signature(MLP).parameters["kwargs"].annotation == int | float


def test_gum_mlp_expects_in_features_parameter_as_int() -> None:
    assert issubclass(signature(MLP).parameters["in_features"].annotation, int)


def test_gum_mlp_expects_out_features_parameter() -> None:
    assert "out_features" in signature(MLP).parameters


def test_gum_mlp_expects_out_features_parameter_as_int_list() -> None:
    assert signature(MLP).parameters["out_features"].annotation == list[int]


def test_gum_mlp_expects_activation_module_parameter() -> None:
    assert "activation_module" in signature(MLP).parameters


def test_gum_mlp_expects_activation_module_parameter_as_type_of_module() -> None:
    assert signature(MLP).parameters["activation_module"].annotation == Type[Module]


@given(mlps())
def test_gum_mlp_is_mlp(
    mlp: MLP,
) -> None:
    assert isinstance(mlp, MLP)


@given(mlps())
def test_gum_mlp_has_attribute_activation_module(mlp: MLP) -> None:
    assert hasattr(mlp, "activation_module")


@given(mlps())
def test_gum_mlp_has_subclass_attribute_activation_module(mlp: MLP) -> None:
    assert issubclass(mlp.activation_module, Module)


@given(
    hst.integers(min_value=1, max_value=10),
    hst.lists(hst.integers(min_value=1, max_value=10), min_size=1, max_size=10),
    betas_or_thresholds(),
)
def test_gum_mlp_gum_softplus_can_be_instantiated_with_keyword_argument_beta(
    in_features: int, out_features: list[int], beta: int
) -> None:
    assert_equal(MLP(in_features, out_features, GUMSoftplus, beta=beta)[1].beta, beta)


@given(
    hst.integers(min_value=1, max_value=10),
    hst.lists(hst.integers(min_value=1, max_value=10), min_size=1, max_size=10),
    betas_or_thresholds(),
)
def test_gum_mlp_gum_softplus_can_be_instantiated_with_keyword_argument_threshold(
    in_features: int, out_features: list[int], threshold: int
) -> None:
    assert_equal(
        MLP(in_features, out_features, GUMSoftplus, threshold=threshold)[1].threshold,
        threshold,
    )


@given(
    hst.integers(min_value=1, max_value=10),
    hst.lists(hst.integers(min_value=1, max_value=10), min_size=1, max_size=10),
    betas_or_thresholds(),
    betas_or_thresholds(),
)
def test_gum_mlp_gum_softplus_init_with_positional_argument_for_threshold(
    in_features: int, out_features: list[int], beta: int, threshold: int
) -> None:
    gum_mlp = MLP(in_features, out_features, GUMSoftplus, beta, threshold)
    assert_equal(gum_mlp[1].threshold, threshold)


@given(mlps(in_dimen=5))
def test_init_mlp_input_layer_as_specified(
    mlp: MLP,
) -> None:
    assert_equal(next(mlp.children()).in_features, 5)


@given(mlps(n_hidden_channels=3))
def test_init_mlp_number_of_layers_as_specified(mlp: MLP) -> None:
    assert_equal(len(mlp), 2 * 3)


@given(mlps())
def test_init_mlp_each_pair_of_layers_is_linear_or_uncertain_linear_and_fix_activation(
    mlp: MLP,
) -> None:
    layer_iter = mlp.children()
    while first_layer_in_a_pair := next(layer_iter, None):
        assert isinstance(first_layer_in_a_pair, GUMLinear) or isinstance(
            first_layer_in_a_pair, Linear
        )
        second_layer_in_a_pair = next(layer_iter)
        assert isinstance(second_layer_in_a_pair, mlp.activation_module)


@given(mlps(uncertain_inputs_exclusively=True))
def test_init_gum_mlp_each_pair_of_layers_is_uncertain_linear_and_uncertain_other(
    gum_mlp: MLP,
) -> None:
    layer_iter = gum_mlp.children()
    while first_layer_in_a_pair := next(layer_iter, None):
        assert isinstance(first_layer_in_a_pair, GUMLinear)
        second_layer_in_a_pair = next(layer_iter)
        assert isinstance(second_layer_in_a_pair, gum_mlp.activation_module)


@given(mlps(activation=hst.sampled_from((Softplus, QuadLU, Sigmoid))))
def test_init_mlp_each_pair_of_layers_is_linear_and_other(
    mlp: MLP,
) -> None:
    layer_iter = mlp.children()
    while first_layer_in_a_pair := next(layer_iter, None):
        assert isinstance(first_layer_in_a_pair, Linear)
        second_layer_in_a_pair = next(layer_iter)
        assert isinstance(second_layer_in_a_pair, mlp.activation_module)


@given(mlps(uncertain_inputs_exclusively=True))
def test_init_gum_mlp_each_pair_of_layers_starts_with_uncertain_linear(
    mlp: MLP,
) -> None:
    layer_iter = mlp.children()
    while first_layer_in_a_pair := next(layer_iter, None):
        assert isinstance(first_layer_in_a_pair, GUMLinear)
        next(layer_iter)


def test_mlp_has_attribute_forward() -> None:
    assert hasattr(MLP, "forward")


def test_mlp_forward_expects_two_parameters() -> None:
    assert_equal(len(signature(MLP.forward).parameters), 2)


@given(uncertain_tensors(length=8), mlps(in_dimen=8, uncertain_inputs_exclusively=True))
@settings(deadline=None)
def test_gum_mlp_outputs_uncertain_tensor_for_gum_activations(
    uncertain_values: UncertainTensor,
    gum_mlp: MLP,
) -> None:
    assert isinstance(gum_mlp(uncertain_values), UncertainTensor)


@given(
    hst.lists(hst.integers(min_value=1, max_value=10), min_size=1, max_size=10),
    uncertain_tensors(),
    hst.sampled_from((GUMSoftplus, GUMQuadLU)),
)
@settings(deadline=None)
def test_mlp_correct_output_dimension_of_values(
    out_dimens: list[int], uncertain_values: UncertainTensor, activation: type[Module]
) -> None:
    mlp = MLP(len(uncertain_values.values), out_dimens, activation)
    assert_equal(len(mlp(uncertain_values).values), out_dimens[-1])


@given(tensors(length=7), mlps(in_dimen=7))
def test_mlp_outputs_the_correct_type_for_activations(values: Tensor, mlp: MLP) -> None:
    if signature(mlp[0].forward).return_annotation is UncertainTensor:
        assert isinstance(mlp(UncertainTensor(values, None)), UncertainTensor)
    else:
        assert isinstance(mlp(values), Tensor)


@given(
    hst.lists(hst.integers(min_value=1, max_value=10), min_size=1, max_size=10),
    uncertain_tensors(),
    hst.sampled_from((GUMSoftplus, GUMQuadLU)),
)
def test_gum_mlp_correct_output_dimension_of_uncertainties(
    out_dimens: list[int], uncertain_values: UncertainTensor, activation: type[Module]
) -> None:
    mlp = MLP(len(uncertain_values.values), out_dimens, activation)
    assert_equal(
        len(mlp(uncertain_values)[1]),
        out_dimens[-1],
    )


@given(
    hst.lists(hst.integers(min_value=1, max_value=10), min_size=1, max_size=10),
    hst.sampled_from((GUMSoftplus, GUMQuadLU)),
)
def test_mlp_correct_output_dimension_of_all_layers(
    out_dimens: list[int], activation: type[Module]
) -> None:
    mlp = MLP(4, out_dimens, activation)
    linear_layers_iter = islice(mlp.children(), None, None, 2)
    out_dimens_iter = iter(out_dimens)
    for linear_layer in linear_layers_iter:
        assert_equal(
            linear_layer.out_features,
            next(out_dimens_iter),
        )


@given(
    tensors(length=9),
    mlps(
        in_dimen=9,
        n_hidden_channels=1,
        out_channels=9,
        activation=hst.just(GUMQuadLU),
    ),
)
def test_gum_mlp_is_correct_for_identity_matrix_product(
    uncertain_quadlu_instance: GUMQuadLU,
    values: Tensor,
    gum_mlp: MLP,
) -> None:
    gum_mlp[0].weight.data.zero_()
    gum_mlp[0].weight.data.fill_diagonal_(1.0)
    assert_close(
        gum_mlp(UncertainTensor(values, None)),
        uncertain_quadlu_instance(
            UncertainTensor(values @ torch.eye(9) + gum_mlp[0].bias, None)
        ),
        equal_nan=True,
    )
