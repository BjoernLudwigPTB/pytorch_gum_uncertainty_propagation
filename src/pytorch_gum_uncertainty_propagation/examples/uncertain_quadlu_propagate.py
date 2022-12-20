"""Does the actual propagation to be profiled for MLPs equipped with QuadLU"""

__all__ = ["assemble_pipeline"]

import torch
from torch.autograd.profiler import profile

from pytorch_gum_uncertainty_propagation.modules import UncertainQuadLUMLP
from pytorch_gum_uncertainty_propagation.uncertainties import (
    UncertainTensor,
)
from pytorch_gum_uncertainty_propagation.zema_dataset import (
    convert_zema_std_uncertainties_into_synthetic_full_cov_matrices,
)


def assemble_pipeline(
    n_samples: int = 1, uncertain_values: UncertainTensor | None = None
) -> profile:
    """Propagate data through an MLP equipped with UncertainQuadLU activation"""
    if uncertain_values is None:
        uncertain_values = (
            convert_zema_std_uncertainties_into_synthetic_full_cov_matrices(n_samples)
        )
    assert uncertain_values.uncertainties is not None
    uncertain_quadlu_mlp = _instantiate_uncertain_quadlu_mlp(
        uncertain_values.values.shape[1],
        _construct_partition(uncertain_values.values.shape[1]),
    )
    for uncertain_value in zip(uncertain_values.values, uncertain_values.uncertainties):
        with profile(with_stack=True, enabled=False) as profiler:
            propagated = uncertain_quadlu_mlp(UncertainTensor(*uncertain_value))
        print(f"propagated and received: \n" f"{propagated}")
    return profiler


def _instantiate_uncertain_quadlu_mlp(
    in_features: int, out_features: list[int]
) -> UncertainQuadLUMLP:
    """Create an instance of an MLP equipped with UncertainQuadLU activation"""
    return UncertainQuadLUMLP(in_features, out_features)


def _construct_partition(in_features: int) -> list[int]:
    """Construct partition of each 0.75 times smaller sections"""
    partition = {in_features}
    while in_features > 2 and len(partition) < 5:
        partition.add(in_features := 3 * in_features // 4)
    return list(sorted(partition, reverse=True))


if __name__ == "__main__":
    profiles = assemble_pipeline(100)
    print(
        profiles.key_averages(group_by_stack_n=2).table(
            sort_by="self_cpu_time_total", row_limit=7
        )
    )
    profiles.export_chrome_trace(path="trace.json")
