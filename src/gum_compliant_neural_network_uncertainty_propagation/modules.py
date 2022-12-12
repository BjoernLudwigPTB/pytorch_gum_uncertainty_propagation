"""Contains the custom activation functions based on QuadLU for now"""

__all__ = ["QuadLU", "QUADLU_ALPHA_DEFAULT", "QuadLUMLP", "UncertainQuadLU"]

from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module, ModuleList, Sequential
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

    def __init__(self, alpha: Parameter = QUADLU_ALPHA_DEFAULT, inplace: bool = False):
        """QuadLU activation function with parameter alpha"""
        super().__init__()
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
        inplace : bool
            can optionally do the operation in-place. Default: ``False``
        """

    QUADLU_ALPHA_DEFAULT: Parameter = QUADLU_ALPHA_DEFAULT

    def __init__(self, alpha: Parameter = QUADLU_ALPHA_DEFAULT, inplace: bool = False):
        """Parametrized QuadLU activation function and uncertainty propagation"""
        super().__init__()
        self._two_alpha = 2 * alpha
        self._quadlu = QuadLU(alpha, inplace)

    def forward(
        self, values: Tensor, uncertainties: Optional[Tensor] = None
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Forward pass of UncertainQuadLU"""
        if uncertainties is None:
            return self._quadlu(values), uncertainties
        first_derivs = (
            uncertainties if self._inplace else torch.zeros_like(uncertainties)
        )
        less_or_equal_mask = values <= -self._quadlu._alpha  # pylint: disable=W0212
        greater_or_equal_mask = values >= self._quadlu._alpha  # pylint: disable=W0212
        first_derivs[greater_or_equal_mask] = (
            4.0 * self._quadlu._alpha  # pylint: disable=W0212
        )
        in_between_mask = ~(less_or_equal_mask | greater_or_equal_mask)
        first_derivs[in_between_mask] = 2 * values + self._two_alpha
        return self._quadlu(values), torch.square(first_derivs) * uncertainties

    @property
    def _alpha(self) -> Parameter:
        return self._quadlu._alpha  # it is still private, pylint: disable=W0212

    @property
    def _inplace(self) -> bool:
        return self._quadlu._inplace  # it is still private, pylint: disable=W0212


class QuadLUMLP(Sequential):
    """This block implements the multi-layer perceptron (MLP) with QuadLU activation

    The implementation is heavily based on the module :class:`~torchvision.ops.MLP`.
    For each specified output dimension a combination of a :class:`~torch.nn.Linear` and
    the :class:`~gum_compliant_neural_network_uncertainty_propagation.modules.QuadLU`
    activation function is added.

    Parameters
    ----------
    in_channels : int
        number of channels of the input
    out_features : list[int]
        the hidden and output layers' dimensions
    """

    def __init__(
        self,
        in_channels: int,
        out_features: list[int],
    ):
        layers = ModuleList()
        in_dimen = in_channels
        for out_dimen in out_features:
            layers.append(torch.nn.Linear(in_dimen, out_dimen, dtype=torch.double))
            layers.append(QuadLU())
            in_dimen = out_dimen
        super().__init__(*layers)
