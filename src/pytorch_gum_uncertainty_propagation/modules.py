"""Contains the custom activation functions based on QuadLU for now"""

__all__ = [
    "QuadLU",
    "QUADLU_ALPHA_DEFAULT",
    "QuadLUMLP",
    "UncertainLinear",
    "UncertainQuadLU",
    "UncertainQuadLUMLP",
]

import torch
from torch import Tensor
from torch.autograd import profiler
from torch.nn import Linear, Module, ModuleList, Sequential
from torch.nn.parameter import Parameter

from pytorch_gum_uncertainty_propagation.functionals import (
    quadlu,
    QUADLU_ALPHA_DEFAULT,
)
from pytorch_gum_uncertainty_propagation.uncertainties import (
    UncertainTensor,
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
        alpha : float, optional
            trainable, non-negative parameter of QuadLU activation function, defaults to
            0.25
        """

    QUADLU_ALPHA_DEFAULT: Parameter = QUADLU_ALPHA_DEFAULT

    def __init__(self, alpha: Parameter = QUADLU_ALPHA_DEFAULT):
        """Parametrized QuadLU activation function and uncertainty propagation"""
        super().__init__()
        self._quadlu = QuadLU(alpha)

    def forward(self, uncertain_values: UncertainTensor) -> UncertainTensor:
        """Forward pass of UncertainQuadLU"""
        with profiler.record_function("UNCERTAINQUADLU PASS"):
            if uncertain_values.uncertainties is None:
                return UncertainTensor(
                    self._quadlu(uncertain_values.values),
                    uncertain_values.uncertainties,
                )
            first_derivs = torch.zeros_like(uncertain_values.values)
            less_or_equal_mask = (
                uncertain_values.values <= -self._quadlu._alpha  # pylint: disable=W0212
            )
            greater_or_equal_mask = (
                uncertain_values.values >= self._quadlu._alpha  # pylint: disable=W0212
            )
            first_derivs[greater_or_equal_mask] = (
                4.0 * self._quadlu._alpha  # pylint: disable=W0212
            )
            in_between_mask = ~(less_or_equal_mask | greater_or_equal_mask)
            first_derivs[in_between_mask] = 2.0 * (
                uncertain_values.values[in_between_mask]
                + self._quadlu._alpha  # pylint: disable=W0212
            )
            return UncertainTensor(
                self._quadlu(uncertain_values.values),
                first_derivs
                * uncertain_values.uncertainties
                * first_derivs.unsqueeze(1),
            )

    @property
    def _alpha(self) -> Parameter:
        """The parameter alpha of the activation function"""
        return self._quadlu._alpha  # it is still private, pylint: disable=W0212


class UncertainLinear(Module):
    """Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    This module supports TensorFloat32.

    On certain ROCm devices, when using float16 inputs this module will use different
    precision for backward.

    Parameters
    ----------
    in_features : int
        size of each input sample
    out_features : int
        size of each output sample
    bias : bool
        If set to False, the layer will not learn an additive bias.
        Default: ``True``
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """A Linear layer with uncertainty propagation"""
        super().__init__()
        self._linear = Linear(in_features, out_features, bias=bias)

    def forward(self, uncertain_values: UncertainTensor) -> UncertainTensor:
        """Forward pass of UncertainLinear"""
        with profiler.record_function("UNCERTAINLINEAR PASS"):
            return UncertainTensor(
                self._linear.forward(uncertain_values.values),
                self.weight @ uncertain_values.uncertainties @ self.weight.T
                if uncertain_values.uncertainties is not None
                else None,
            )

    @property
    def in_features(self) -> int:
        """Size of each input sample"""
        return self._linear.in_features

    @property
    def out_features(self) -> int:
        """Size of each output sample"""
        return self._linear.out_features

    @property
    def bias(self) -> Tensor:
        r"""The learnable additive bias of the module of shape ``(out_features)``

        If ``bias`` is ``True``, the values are initialized from :math:`\mathcal{U}
        (-\sqrt{k}, \sqrt{k})`, where :math:`k=\frac{1}{\text{in_features}}`.(
        """
        return self._linear.bias

    @property
    def weight(self) -> Tensor:
        r"""The learnable weights of the module of shape ``(out_features, in_features)``

        The values are initialized from :math:`\mathcal{U} (-\sqrt{k}, \sqrt{k})`,
        where :math:`k=\frac{1}{\text{in_features}}`.
        """
        return self._linear.weight


class QuadLUMLP(Sequential):
    """This implements the multi-layer perceptron (MLP) with QuadLU activation

    The implementation is heavily based on the module :class:`~torchvision.ops.MLP`.
    For each specified output dimension a combination of a :class:`~torch.nn.Linear` and
    the :class:`~pytorch_gum_uncertainty_propagation.modules.QuadLU`
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
        """An MLP consisting of linear layers and QuadLU activation function"""
        layers = ModuleList()
        in_dimen = in_channels
        for out_dimen in out_features:
            layers.append(torch.nn.Linear(in_dimen, out_dimen, dtype=torch.double))
            layers.append(QuadLU())
            in_dimen = out_dimen
        super().__init__(*layers)


class UncertainQuadLUMLP(Sequential):
    """This implements the multi-layer perceptron (MLP) with UncertainQuadLU activation

    The implementation is heavily based on the module :class:`~torchvision.ops.MLP`.
    For each specified output dimension a combination of a
    :class:`UncertainLinear` and the :class:`UncertainQuadLU` activation function is
    added.

    Parameters
    ----------
    in_channels : int
        number of channels of the input
    out_features : list[int]
        the hidden and output layers' dimensions
    """

    def __init__(self, in_channels: int, out_features: list[int]) -> None:
        """An MLP consisting of UncertainLinear QuadLU layers"""
        super().__init__()
        layers = ModuleList()
        for out_dimen in out_features:
            layers.append(UncertainLinear(in_channels, out_dimen))
            layers.append(UncertainQuadLU())
            in_channels = out_dimen
        super().__init__(*layers)
