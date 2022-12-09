"""Contains the functional version of the custom activation function QuadLU for now"""

__all__ = ["quadlu", "QUADLU_ALPHA_DEFAULT"]

import torch
from torch import Tensor
from torch.nn.parameter import Parameter

QUADLU_ALPHA_DEFAULT = Parameter(torch.tensor(0.25))


def quadlu(values: Tensor, alpha: Parameter = QUADLU_ALPHA_DEFAULT) -> Tensor:
    r"""Applies QuadLU element-wise

    See :class:`~gum_compliant_neural_network_uncertainty_propagation.modules.QuadLU`
    for more details.
    """
    result_tensor = torch.empty_like(values)
    less_or_equal_mask = values <= -alpha
    result_tensor[less_or_equal_mask] = 0.0
    greater_or_equal_mask = values >= alpha
    result_tensor[greater_or_equal_mask] = 4.0 * alpha * values[greater_or_equal_mask]
    in_between_mask = ~(less_or_equal_mask | greater_or_equal_mask)
    result_tensor[in_between_mask] = (values[in_between_mask] + alpha) ** 2
    return result_tensor
