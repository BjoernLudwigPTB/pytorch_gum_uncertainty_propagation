"""Fixture for all tests related to the module 'modules'"""
from typing import cast, NamedTuple

import pytest
from hypothesis import strategies as hst
from hypothesis.strategies import composite, DrawFn, SearchStrategy
from torch import Tensor

from pytorch_gum_uncertainty_propagation.modules import (
    QuadLU,
    GUMLinear,
    GUMQuadLU,
)
from pytorch_gum_uncertainty_propagation.uncertainties import (
    cov_matrix_from_std_uncertainties,
    UncertainTensor,
)
from ..conftest import tensors


@pytest.fixture(scope="session")
def quadlu_instance() -> QuadLU:
    return QuadLU()


@pytest.fixture(scope="session")
def uncertain_quadlu_instance() -> GUMQuadLU:
    return GUMQuadLU()


@composite
def uncertain_linears(
    draw: DrawFn,
    in_features: int | None = None,
    out_features: int | None = None,
    bias: bool = True,
) -> SearchStrategy[GUMLinear]:
    in_features = (
        in_features
        if in_features is not None
        else draw(hst.integers(min_value=1, max_value=10))
    )
    out_features = (
        out_features
        if out_features is not None
        else draw(hst.integers(min_value=1, max_value=10))
    )
    return cast(
        SearchStrategy[GUMLinear],
        GUMLinear(in_features, out_features, bias=bias),
    )


class UncertainTensorForLinear(NamedTuple):
    uncertain_values: UncertainTensor
    uncertain_linear: GUMLinear


@composite
def values_uncertainties_and_uncertain_linears(
    draw: DrawFn, greater_than: float = -1e2, less_than: float = 1e2
) -> SearchStrategy[UncertainTensorForLinear]:
    values: Tensor = cast(
        Tensor,
        draw(tensors(elements_min=greater_than, elements_max=less_than)),
    )
    std_uncertainties = cast(
        Tensor,
        draw(
            tensors(
                elements_min=values.abs().min().data.item() * 1e-3,
                elements_max=values.abs().min().data.item() * 1e2,
                length=len(values),
            )
        ),
    )
    cov_matrix = cov_matrix_from_std_uncertainties(std_uncertainties)
    return cast(
        SearchStrategy[UncertainTensorForLinear],
        UncertainTensorForLinear(
            UncertainTensor(values, cov_matrix),
            GUMLinear(len(values), draw(hst.integers(min_value=1, max_value=10))),
        ),
    )


@composite
def betas_or_thresholds(draw: DrawFn) -> SearchStrategy[int]:
    return cast(SearchStrategy[int], draw(hst.integers(min_value=1, max_value=20)))
