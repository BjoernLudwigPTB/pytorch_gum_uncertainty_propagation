"""Test the class QuadLUMLP"""
from typing import cast

import torch
from hypothesis import given, strategies as hst
from hypothesis.strategies import composite, DrawFn, SearchStrategy
from numpy.testing import assert_equal
from torch import Tensor
from torch.testing import assert_close  # type: ignore[attr-defined]

from pytorch_gum_uncertainty_propagation import modules
from pytorch_gum_uncertainty_propagation.modules import (
    MLP,
    QuadLU,
    QuadLUMLP,
)
from ..conftest import tensors


@composite
def quadlu_mlps(
    draw: DrawFn,
    in_dimen: int | None = None,
    n_hidden_channels: int | None = None,
    out_channels: int | None = None,
) -> SearchStrategy[QuadLUMLP]:
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
    return cast(SearchStrategy[QuadLUMLP], QuadLUMLP(in_dimen, hidden_and_out_dimens))


def test_modules_all_contains_quadlu_mlp() -> None:
    assert QuadLUMLP.__name__ in modules.__all__


def test_quadlu_mlp_is_subclass_of_nn_module() -> None:
    assert issubclass(QuadLUMLP, MLP)


def test_quadlu_mlp_has_docstring() -> None:
    assert QuadLUMLP.__doc__ is not None


@given(quadlu_mlps())
def test_init_quadlu_mlp(quadlu_mlp: QuadLUMLP) -> None:
    assert isinstance(quadlu_mlp, QuadLUMLP)


@given(quadlu_mlps(in_dimen=5))
def test_init_quadlu_mlp_input_layer_as_specified(quadlu_mlp: QuadLUMLP) -> None:
    assert_equal(next(quadlu_mlp.children()).in_features, 5)


@given(quadlu_mlps(n_hidden_channels=3))
def test_init_quadlu_mlp_input_dimension_as_specified(quadlu_mlp: QuadLUMLP) -> None:
    assert_equal(len(quadlu_mlp), 2 * 3)


@given(tensors(length=8), quadlu_mlps(in_dimen=8))
def test_quadlu_mlp_outputs_tensor(values: Tensor, quadlu_mlp: QuadLUMLP) -> None:
    assert isinstance(quadlu_mlp(values), Tensor)


@given(tensors(length=6), quadlu_mlps(in_dimen=6, out_channels=3))
def test_quadlu_mlp_correct_output_dimension(
    values: Tensor, quadlu_mlp: QuadLUMLP
) -> None:
    assert_equal(len(quadlu_mlp(values)), 3)


@given(
    tensors(length=9),
    quadlu_mlps(in_dimen=9, n_hidden_channels=1, out_channels=9),
)
def test_quadlu_mlp_is_correct_for_identity_matrix_product(
    quadlu_instance: QuadLU, values: Tensor, quadlu_mlp: QuadLUMLP
) -> None:
    quadlu_mlp[0].weight.data.zero_()
    quadlu_mlp[0].weight.data.fill_diagonal_(1.0)
    assert_close(
        quadlu_mlp(values),
        quadlu_instance(values @ torch.eye(9, dtype=torch.double) + quadlu_mlp[0].bias),
        equal_nan=True,
    )
