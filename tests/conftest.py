from typing import cast

from hypothesis import strategies as hst
from hypothesis.extra import numpy as hnp
from hypothesis.strategies import composite, DrawFn, SearchStrategy
from torch import tensor, Tensor
from torch.nn.parameter import Parameter


@composite
def alphas(
    draw: DrawFn,
    min_value: float = 0.0,
    exclude_min: bool = True,
    max_value: float = 10,
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
    draw: DrawFn, elements_min=None, elements_max=None
) -> SearchStrategy[Tensor]:
    return cast(
        SearchStrategy[Tensor],
        tensor(
            draw(
                hnp.arrays(
                    dtype=float,
                    shape=hnp.array_shapes(
                        min_dims=1,
                        max_dims=1,
                        min_side=1,
                        max_side=10,
                    ),
                    elements=hst.floats(min_value=elements_min, max_value=elements_max),
                ),
            )
        ),
    )
