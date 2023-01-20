"""Does the actual propagation to be profiled for MLPs equipped with GUMQuadLU"""

__all__ = ["assemble_pipeline"]

import datetime
from itertools import repeat
from math import ceil
from typing import Any, Iterable, Type

import torch
from torch import Tensor
from torch.autograd.profiler import profile
from torch.nn import Module

from pytorch_gum_uncertainty_propagation.examples.zema_dataset import (
    convert_zema_std_uncertainties_into_synthetic_full_cov_matrices,
)
from pytorch_gum_uncertainty_propagation.modules import (
    GUMQuadLUMLP,
    GUMSigmoidMLP,
    GUMSoftplusMLP,
)
from pytorch_gum_uncertainty_propagation.uncertainties import (
    UncertainTensor,
)


def assemble_pipeline(
    generic_mlp_module: Type[Module],
    size_scaler: int = 1,
    idx_start: int = 0,
    depth: int = 1,
    set_uncertainties_to_none: bool = False,
) -> Any:
    """Propagate data through an MLP equipped with GUMQuadLU activation"""
    torch.set_default_dtype(torch.double)  # type: ignore[no-untyped-call]
    input_values = convert_zema_std_uncertainties_into_synthetic_full_cov_matrices(
        n_samples=1, size_scaler=size_scaler, normalize=True, idx_start=idx_start
    )
    assert input_values.uncertainties is not None
    if set_uncertainties_to_none:
        uncertainties: Tensor | Iterable[None] = repeat(None)
    else:
        uncertainties = input_values.uncertainties
    if size_scaler == 1:
        out_features = 11 - depth
    elif (max_outputs := size_scaler * 11 - depth) >= 100:
        out_features = 100
    else:
        out_features = max_outputs
    assert 11 - depth <= out_features <= 100
    mlp = generic_mlp_module(
        input_values.values.shape[1],
        _construct_out_features_counts(
            input_values.values.shape[1], out_features, depth
        ),
    )  # type: ignore[call-arg]
    for uncertain_value in zip(input_values.values, uncertainties):
        with profile(with_stack=True) as profiler:  # type: ignore[no-untyped-call]
            mlp(UncertainTensor(*uncertain_value))
    return profiler


def _construct_out_features_counts(
    in_features: int, out_features: int = 2, depth: int = 1
) -> list[int]:
    """Construct network architecture with desired depth for parameter generation"""
    if depth == 1:
        return [out_features]
    assert in_features > out_features
    assert (in_features - out_features) / depth >= 1.0
    partition = {out_features}
    while len(partition) < depth:
        step = (in_features - out_features) / (depth - len(partition) + 1)
        partition.add(in_features := ceil(in_features - step))
    assert len(partition) == depth
    assert min(partition) == out_features
    return list(sorted(partition, reverse=True))


def iterate_over_activations_and_architectures(
    depths: tuple[int, ...], size_scalers: tuple[int, ...]
) -> None:
    """Iterate over GUM modules for hard coded architectures and ZeMA sample sizes"""
    for MLPModule in (GUMSoftplusMLP, GUMQuadLUMLP, GUMSigmoidMLP):
        for layers_additional_to_input in depths:
            for samples_per_sensor in size_scalers:
                for set_to_none in (False, True):
                    with open("timings.txt", "a", encoding="utf-8") as timings_file:
                        timings_file.write(
                            f"\n======================================================="
                            f"====================================================\n"
                            f"Timing {MLPModule.__name__} for {samples_per_sensor * 11}"
                            f" inputs "
                            f"{'without' if set_to_none else 'with'} "
                            f"uncertainties and "
                            f"{layers_additional_to_input} "
                            f"{'layers' if layers_additional_to_input > 1 else 'layer'}"
                            f"\n======================================================="
                            f"====================================================\n"
                        )
                    profiles = assemble_pipeline(
                        MLPModule,
                        samples_per_sensor,
                        depth=layers_additional_to_input,
                    )
                    with open("timings.txt", "a", encoding="utf-8") as timings_file:
                        timings_file.write(
                            profiles.key_averages(group_by_stack_n=2).table(
                                sort_by="cpu_time_total", row_limit=15
                            )
                        )
                    profiles.export_chrome_trace(
                        path=f""
                        f"{datetime.datetime.now().isoformat('_','hours')}_"
                        f"{MLPModule.__name__}_{samples_per_sensor * 11}_inputs_"
                        f"{layers_additional_to_input}_layers_trace.json"
                    )


if __name__ == "__main__":
    iterate_over_activations_and_architectures((1, 3, 5, 8), (1, 10, 100, 1000, 2000))
