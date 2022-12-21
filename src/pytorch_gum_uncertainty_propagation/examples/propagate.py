"""Does the actual propagation to be profiled for MLPs equipped with GUMQuadLU"""

__all__ = ["assemble_pipeline"]

import datetime
from typing import Any, Type

import torch
from torch import Tensor
from torch.autograd.profiler import profile
from torch.nn import Module

from pytorch_gum_uncertainty_propagation.modules import (
    GUMQuadLUMLP,
    GUMSigmoidMLP,
    GUMSoftplusMLP,
    QuadLUMLP,
)
from pytorch_gum_uncertainty_propagation.uncertainties import (
    UncertainTensor,
)
from pytorch_gum_uncertainty_propagation.zema_dataset import (
    convert_zema_std_uncertainties_into_synthetic_full_cov_matrices,
    provide_zema_samples,
)


def assemble_pipeline(
    generic_mlp_module: Type[Module],
    n_samples: int = 1,
    input_values: UncertainTensor | Tensor | None = None,
) -> Any:
    """Propagate data through an MLP equipped with GUMQuadLU activation"""
    torch.set_default_dtype(torch.double)  # type: ignore[no-untyped-call]
    if input_values is None:
        if "GUM" in generic_mlp_module.__name__:
            input_values = (
                convert_zema_std_uncertainties_into_synthetic_full_cov_matrices(
                    n_samples
                )
            )
            assert input_values.uncertainties is not None
            mlp = generic_mlp_module(
                input_values.values.shape[1],
                _construct_partition(input_values.values.shape[1]),
            )  # type: ignore[call-arg]
            for uncertain_value in zip(input_values.values, input_values.uncertainties):
                with profile(
                    with_stack=True
                ) as profiler:  # type: ignore[no-untyped-call]
                    mlp(UncertainTensor(*uncertain_value))
        else:
            input_values = provide_zema_samples(n_samples).values
            mlp = generic_mlp_module(
                input_values.shape[1], _construct_partition(input_values.shape[1])
            )  # type: ignore[call-arg]
            for value in input_values:
                with profile(
                    with_stack=True
                ) as profiler:  # type: ignore[no-untyped-call]
                    mlp(value)

    return profiler


def _construct_partition(in_features: int) -> list[int]:
    """Construct partition of each 0.75 times smaller sections"""
    partition = {in_features}
    while in_features > 2 and len(partition) < 5:
        partition.add(in_features := 3 * in_features // 4)
    return list(sorted(partition, reverse=True))


if __name__ == "__main__":
    for mlp_module in (GUMSoftplusMLP, GUMSigmoidMLP, GUMQuadLUMLP, QuadLUMLP):
        profiles = assemble_pipeline(GUMSoftplusMLP, 10)
        print(
            profiles.key_averages(group_by_stack_n=2).table(
                sort_by="self_cpu_time_total", row_limit=15
            )
        )
        profiles.export_chrome_trace(
            path=f""
            f"{datetime.datetime.now().isoformat('_','hours')}_"
            f"{mlp_module.__name__}_trace.json"
        )
