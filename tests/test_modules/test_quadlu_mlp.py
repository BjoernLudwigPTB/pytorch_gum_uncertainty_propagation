"""Test the class QuadLUMLP"""
from typing import cast, Optional

from hypothesis import given, strategies as hst
from hypothesis.strategies import composite, DrawFn, SearchStrategy
from numpy.testing import assert_equal
from torch import Tensor
from torch.nn import Sequential
from torch.testing import assert_close  # type: ignore[attr-defined]

from gum_compliant_neural_network_uncertainty_propagation import modules
from gum_compliant_neural_network_uncertainty_propagation.modules import (
    QuadLUMLP,
)
from ..conftest import tensors


@composite
def quadlu_mlps(
    draw: DrawFn,
    in_channels: Optional[int] = None,
    n_hidden_channels: Optional[int] = None,
    out_channels: Optional[int] = None,
) -> SearchStrategy[QuadLUMLP]:
    dimen_strategy = hst.integers(min_value=1, max_value=100)
    if in_channels is None:
        in_channels = draw(dimen_strategy)
    if n_hidden_channels is None:
        n_hidden_channels = draw(dimen_strategy)
    hidden_channel_dimens = []
    for _ in range(n_hidden_channels - 1):
        hidden_channel_dimens.append(draw(dimen_strategy))
    if out_channels is None:
        hidden_channel_dimens.append(draw(dimen_strategy))
    else:
        hidden_channel_dimens.append(out_channels)
    return cast(
        SearchStrategy[QuadLUMLP], QuadLUMLP(in_channels, hidden_channel_dimens)
    )


def test_modules_all_contains_quadlu_mlp() -> None:
    assert QuadLUMLP.__name__ in modules.__all__


def test_quadlu_mlp_is_subclass_of_nn_module() -> None:
    assert issubclass(QuadLUMLP, Sequential)


def test_quadlu_mlp_has_docstring() -> None:
    assert QuadLUMLP.__doc__ is not None


@given(quadlu_mlps())
def test_init_quadlu_mlp(quadlu_mlp: QuadLUMLP) -> None:
    assert isinstance(quadlu_mlp, QuadLUMLP)


@given(quadlu_mlps(in_channels=5))
def test_init_quadlu_mlp_input_layer_as_specified(quadlu_mlp: QuadLUMLP) -> None:
    assert_equal(next(quadlu_mlp.children()).in_features, 5)


@given(quadlu_mlps(n_hidden_channels=3))
def test_init_quadlu_mlp_input_dimension_as_specified(quadlu_mlp: QuadLUMLP) -> None:
    assert_equal(len(quadlu_mlp), 2 * 3)


@given(tensors(length=8), quadlu_mlps(in_channels=8))
def test_quadlu_mlp_outputs_tensor(values: Tensor, quadlu_mlp: QuadLUMLP) -> None:
    assert isinstance(quadlu_mlp(values), Tensor)


@given(tensors(length=6), quadlu_mlps(in_channels=6, out_channels=3))
def test_quadlu_mlp_correct_output_dimension(
    values: Tensor, quadlu_mlp: QuadLUMLP
) -> None:
    assert_equal(len(quadlu_mlp(values)), 3)
