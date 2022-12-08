"""Contains the custom activation function QuadLU for now"""

__all__ = ["QuadLU"]

from typing import cast, Optional

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.parameter import Parameter


class QuadLU(Module):
    r"""Implementation of QuadLU activation with parameter alpha

        :math:`\operatorname{QuadLU}_\alpha \colon \mathbb{R} \to \mathbb{R}` is defined
        as

        .. math::

            \operatorname{QuadLU}_\alpha (x) :=
            \begin{cases}
              0, &\quad \text{for } x \leq -\alpha \\
              (x + \alpha)^2, &\quad \text{for } -\alpha < x < \alpha \\
              4\alpha x, &\quad \text{for } x \geq \alpha \\
            \end{cases}

        with :math:`\alpha \in \mathbb{R}_+`.

        Parameters
        ----------
        alpha : float
            trainable, non-negative parameter of QuadLU activation function
        """

    ALPHA_DEFAULT: float = 1.0

    def __init__(self, alpha: Optional[float] = None):
        """QuadLU activation function with parameter alpha"""
        super().__init__()
        if alpha is None:
            self._alpha = Parameter(torch.tensor(self.ALPHA_DEFAULT))
        else:
            self._alpha = Parameter(torch.tensor(alpha))
        self._alpha.requires_grad = True

    def forward(self, x_in: Tensor) -> Tensor:
        """Forward pass of QuadLU"""
        if x_in <= -self._alpha:
            return torch.tensor(0.0)
        if x_in >= self._alpha:
            return 4.0 * self._alpha * x_in
        return cast(Tensor, (x_in + self._alpha) ** 2)
