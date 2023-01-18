"""Test the class GUMQuadLUMLP"""
from inspect import signature
from itertools import islice
from typing import cast

import torch
from hypothesis import given, settings, strategies as hst
from hypothesis.strategies import composite, DrawFn, SearchStrategy
from numpy.testing import assert_equal
from torch.nn import ModuleList
from torch.testing import assert_close  # type: ignore[attr-defined]

from pytorch_gum_uncertainty_propagation import modules
from pytorch_gum_uncertainty_propagation.modules import (
    MLP,
    GUMLinear,
    GUMQuadLU,
    GUMQuadLUMLP,
)
from pytorch_gum_uncertainty_propagation.uncertainties import (
    UncertainTensor,
)
from ..conftest import uncertain_tensors


@composite
def uncertain_quadlu_mlps(
    draw: DrawFn,
    in_dimen: int | None = None,
    n_hidden_channels: int | None = None,
    out_channels: int | None = None,
) -> SearchStrategy[GUMQuadLUMLP]:
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
        SearchStrategy[GUMQuadLUMLP],
        GUMQuadLUMLP(in_dimen, hidden_and_out_dimens),
    )


def test_modules_contains_uncertain_quadlu_mlp() -> None:
    assert hasattr(modules, "GUMQuadLUMLP")


def test_modules_all_contains_uncertain_quadlu_mlp() -> None:
    assert GUMQuadLUMLP.__name__ in modules.__all__


def test_uncertain_quadlu_mlp_is_subclass_of_nn_module() -> None:
    assert issubclass(GUMQuadLUMLP, MLP)


def test_uncertain_quadlu_mlp_has_docstring() -> None:
    assert GUMQuadLUMLP.__doc__ is not None


def test_uncertain_quadlu_mlp_expects_in_features_parameter() -> None:
    assert "in_features" in signature(GUMQuadLUMLP).parameters


def test_uncertain_quadlu_mlp_expects_in_features_parameter_as_int() -> None:
    assert issubclass(signature(GUMQuadLUMLP).parameters["in_features"].annotation, int)


def test_uncertain_quadlu_mlp_expects_out_features_parameter() -> None:
    assert "out_features" in signature(GUMQuadLUMLP).parameters


@given(uncertain_quadlu_mlps())
def test_uncertain_quadlu_mlp_is_uncertain_quadlu_mlp(
    uncertain_quadlu_mlp: GUMQuadLUMLP,
) -> None:
    assert isinstance(uncertain_quadlu_mlp, GUMQuadLUMLP)


@given(uncertain_quadlu_mlps())
def test_uncertain_quadlu_mlp_children_does_not_provide_access_to_module_list(
    uncertain_quadlu_mlp: GUMQuadLUMLP,
) -> None:
    assert type(next(uncertain_quadlu_mlp.children())) is not ModuleList


@given(uncertain_quadlu_mlps())
def test_uncertain_quadlu_mlp_children_provides_access_to_uncertain_linears_and_quadlus(
    uncertain_quadlu_mlp: GUMQuadLUMLP,
) -> None:
    for child in uncertain_quadlu_mlp.children():
        assert type(child) is GUMLinear or type(child) is GUMQuadLU


def test_uncertain_quadlu_mlp_children_has_docstring() -> None:
    assert GUMQuadLUMLP.children.__doc__ is not None


@given(uncertain_quadlu_mlps(in_dimen=5))
def test_init_uncertain_quadlu_mlp_input_layer_as_specified(
    uncertain_quadlu_mlp: GUMQuadLUMLP,
) -> None:
    assert_equal(next(uncertain_quadlu_mlp.children()).in_features, 5)


@given(uncertain_quadlu_mlps(n_hidden_channels=3))
def test_init_uncertain_quadlu_mlp_input_dimension_as_specified(
    uncertain_quadlu_mlp: GUMQuadLUMLP,
) -> None:
    assert_equal(len(uncertain_quadlu_mlp), 2 * 3)


@given(uncertain_quadlu_mlps())
def test_init_uncertain_quadlu_mlp_each_pair_of_layers_is_linear_and_uncertain_quadlu(
    uncertain_quadlu_mlp: GUMQuadLUMLP,
) -> None:
    layer_iter = uncertain_quadlu_mlp.children()
    while first_layer_in_a_pair := next(layer_iter, None):
        assert isinstance(first_layer_in_a_pair, GUMLinear)
        second_layer_in_a_pair = next(layer_iter)
        assert isinstance(second_layer_in_a_pair, GUMQuadLU)


def test_uncertain_quadlu_mlp_has_attribute_forward() -> None:
    assert hasattr(GUMQuadLUMLP, "forward")


def test_uncertain_quadlu_mlp_forward_expects_two_parameters() -> None:
    assert_equal(len(signature(GUMQuadLUMLP.forward).parameters), 2)


@given(uncertain_tensors(length=8), uncertain_quadlu_mlps(in_dimen=8))
@settings(deadline=None)
def test_uncertain_quadlu_mlp_outputs_tuple_tensor(
    values_and_uncertainties: UncertainTensor,
    uncertain_quadlu_mlp: GUMQuadLUMLP,
) -> None:
    assert isinstance(uncertain_quadlu_mlp(values_and_uncertainties), UncertainTensor)


@given(
    hst.lists(hst.integers(min_value=1, max_value=10), min_size=1, max_size=10),
    uncertain_tensors(),
)
def test_uncertain_quadlu_mlp_correct_output_dimension_of_values(
    out_dimens: list[int],
    uncertain_values: UncertainTensor,
) -> None:
    uncertain_quadlu_mlp = GUMQuadLUMLP(len(uncertain_values.values), out_dimens)
    assert_equal(
        len(uncertain_quadlu_mlp(uncertain_values)[0]),
        out_dimens[-1],
    )


@given(
    hst.lists(hst.integers(min_value=1, max_value=10), min_size=1, max_size=10),
    uncertain_tensors(),
)
def test_uncertain_quadlu_mlp_correct_output_dimension_of_uncertainties(
    out_dimens: list[int],
    uncertain_values: UncertainTensor,
) -> None:
    uncertain_quadlu_mlp = GUMQuadLUMLP(len(uncertain_values.values), out_dimens)
    assert_equal(
        len(uncertain_quadlu_mlp(uncertain_values)[1]),
        out_dimens[-1],
    )


@given(hst.lists(hst.integers(min_value=1, max_value=10), min_size=1, max_size=10))
def test_uncertain_quadlu_mlp_correct_output_dimension_of_all_layers(
    out_dimens: list[int],
) -> None:
    uncertain_quadlu_mlp = GUMQuadLUMLP(4, out_dimens)
    linear_layers_iter = islice(uncertain_quadlu_mlp.children(), None, None, 2)
    out_dimens_iter = iter(out_dimens)
    for linear_layer in linear_layers_iter:
        assert_equal(
            linear_layer.out_features,
            next(out_dimens_iter),
        )


@given(
    uncertain_tensors(length=9),
    uncertain_quadlu_mlps(in_dimen=9, n_hidden_channels=1, out_channels=9),
)
def test_uncertain_quadlu_mlp_is_correct_for_identity_matrix_product(
    uncertain_quadlu_instance: GUMQuadLU,
    uncertain_values: UncertainTensor,
    uncertain_quadlu_mlp: GUMQuadLUMLP,
) -> None:
    uncertain_quadlu_mlp[0].weight.data.zero_()
    uncertain_quadlu_mlp[0].weight.data.fill_diagonal_(1.0)
    assert_close(
        uncertain_quadlu_mlp(UncertainTensor(uncertain_values.values, None)),
        uncertain_quadlu_instance(
            UncertainTensor(
                uncertain_values.values @ torch.eye(9) + uncertain_quadlu_mlp[0].bias,
                None,
            )
        ),
    )
