"""Contains the custom activation functions based on QuadLU for now"""

__all__ = ["QuadLU", "QUADLU_ALPHA_DEFAULT", "UncertainQuadLU"]

from typing import Optional

from torch import Tensor
from torch.nn import Module
from torch.nn.parameter import Parameter

from gum_compliant_neural_network_uncertainty_propagation.functionals import (
    quadlu,
    QUADLU_ALPHA_DEFAULT,
)


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
        inplace : bool
            can optionally do the operation in-place. Default: ``False``
        """

    QUADLU_ALPHA_DEFAULT: Parameter = QUADLU_ALPHA_DEFAULT

    def __init__(self, alpha: Optional[Parameter] = None, inplace: bool = False):
        """QuadLU activation function with parameter alpha"""
        super().__init__()
        if alpha is None:
            self._alpha = self.QUADLU_ALPHA_DEFAULT
        else:
            self._alpha = alpha
        self._inplace = inplace

    def forward(self, values: Tensor) -> Tensor:
        """Forward pass of QuadLU"""
        return quadlu(values, self._alpha, self._inplace)


class UncertainQuadLU(Module):
    r"""Implementation of parametrized QuadLU activation with uncertainty propagation

        :math:`\operatorname{QuadLU}_\alpha \colon \mathbb{R} \to \mathbb{R}` is defined
        as

        .. math::

            \operatorname{QuadLU}_\alpha (x) :=
            \begin{cases}
              0, &\quad \text{for } x \leq -\alpha \\
              (x + \alpha)^2, &\quad \text{for } -\alpha < x < \alpha \\
              4\alpha x, &\quad \text{for } x \geq \alpha \\
            \end{cases}

        with :math:`\alpha \in \mathbb{R}_+`. The uncertainty propagation is performed
        as stated in the thesis

        .. math::

            \mathbf{U}_{x^{(\ell)}} = \prod_{i=0}^{\ell-1} \operatorname{diag}
            \frac{\mathrm{d} \operatorname{QuadLU}}{\mathrm{d} z} (z^{(\ell-i)})
            W^{(\ell-i)} \mathbf{U}_{x^{(0)}} \prod_{i=1}^{\ell} {W^{(i)}}^T
            \operatorname{diag} \frac{\mathrm{d} \operatorname{QuadLU}}{\mathrm{d} z}
            (z^{(i)}),

        Parameters
        ----------
        alpha : float
            trainable, non-negative parameter of QuadLU activation function
        """

    QUADLU_ALPHA_DEFAULT: Parameter = QUADLU_ALPHA_DEFAULT

    def __init__(self, alpha: Optional[Parameter] = None):
        """Parametrized QuadLU activation function and uncertainty propagation"""
        super().__init__()
        self._quadlu = QuadLU(alpha)
        self._alpha = self._quadlu._alpha

    def forward(
        self, values: Tensor, uncertainties: Optional[Tensor] = None
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Forward pass of UncertainQuadLU"""
        values = self._quadlu.forward(values)
        return values, uncertainties
