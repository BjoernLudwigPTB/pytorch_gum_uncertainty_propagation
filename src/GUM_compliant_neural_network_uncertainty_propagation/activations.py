"""Contains the custom activation function QuadLU for now"""
import torch
from torch import tensor
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

    ALPHA_DEFAULT: torch.float64 = 1.0

    def __init__(self, alpha: torch.float64 = None):
        """QuadLU activation function with parameter alpha"""
        super(QuadLU, self).__init__()
        if alpha is None:
            self._alpha = Parameter(tensor(self.ALPHA_DEFAULT))
        else:
            self._alpha = Parameter(tensor(alpha))
        self._alpha.requires_grad = True
