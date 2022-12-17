"""Common strategies"""
from typing import cast, Optional

from hypothesis import strategies as hst
from hypothesis.extra import numpy as hnp
from hypothesis.strategies import composite, DrawFn, SearchStrategy
from torch import tensor, Tensor
from torch.nn.parameter import Parameter

from gum_compliant_neural_network_uncertainty_propagation.uncertainties import (
    cov_matrix_from_std_uncertainties,
    UncertainTensor,
)


@composite
def alphas(
    draw: DrawFn,
    min_value: float = 0.0,
    exclude_min: bool = True,
    max_value: float = 10.0,
) -> SearchStrategy[Parameter]:
    return cast(
        SearchStrategy[Parameter],
        Parameter(
            tensor(
                draw(
                    hst.floats(
                        min_value=min_value,
                        exclude_min=exclude_min,
                        max_value=max_value,
                    )
                )
            )
        ),
    )


@composite
def tensors(
    draw: DrawFn,
    length: Optional[int] = None,
    elements_min: Optional[float] = None,
    elements_max: Optional[float] = None,
) -> SearchStrategy[Tensor]:
    """A search strategy for PyTorch tensors

    Parameters
    ----------
    draw : DrawFn
        special function, which can be used just within a test to draw from other
        strategies
    length : optional, float
        the length of the created tensor
    elements_min : optional, float
        the parameter used to set min_value in the elements' float strategy
    elements_max : optional,float
        the parameter used to set max_value in the elements' float strategy

    Returns
    -------
    SearchStrategy[Tensor]
        the search strategy
    """
    if length is None:
        min_side, max_side = 1, 10
    else:
        min_side = max_side = length
    return cast(
        SearchStrategy[Tensor],
        tensor(
            draw(
                hnp.arrays(
                    dtype=float,
                    shape=hnp.array_shapes(
                        min_dims=1,
                        max_dims=1,
                        min_side=min_side,
                        max_side=max_side,
                    ),
                    elements=hst.floats(min_value=elements_min, max_value=elements_max),
                ),
            )
        ),
    )


@composite
def uncertain_tensors(
    draw: DrawFn,
    greater_than: float = -1e2,
    less_than: float = 1e2,
    length: int | None = None,
) -> SearchStrategy[UncertainTensor]:
    values: Tensor = cast(
        Tensor,
        draw(tensors(elements_min=greater_than, elements_max=less_than, length=length)),
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
    cov_matrix = cov_matrix_from_std_uncertainties(std_uncertainties, 0.5, 0.5, 0.5)
    return cast(
        SearchStrategy[UncertainTensor],
        UncertainTensor(values.float(), cov_matrix.float()),
    )
