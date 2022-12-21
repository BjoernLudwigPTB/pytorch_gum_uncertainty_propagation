"""Contains the custom activation functions based on QuadLU for now"""

__all__ = [
    "GUMSigmoid",
    "GUMSigmoidMLP",
    "GUMSoftplus",
    "GUMSoftplusMLP",
    "MLP",
    "QuadLU",
    "QUADLU_ALPHA_DEFAULT",
    "QuadLUMLP",
    "GUMLinear",
    "GUMQuadLU",
    "GUMQuadLUMLP",
]

from inspect import signature
from typing import Type

import torch
from torch import Tensor
from torch.autograd import profiler
from torch.nn import Linear, Module, ModuleList, Sequential, Sigmoid, Softplus
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


class GUMQuadLU(Module):
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
        """Forward pass of GUMQuadLU"""
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


class GUMLinear(Module):
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
        """Forward pass of GUMLinear"""
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


class MLP(Sequential):
    """This implements the multi-layer perceptron (MLP) for any kind of activation

    The implementation is inspired by the module :class:`~torchvision.ops.MLP`.
    For each specified output dimension a combination of a suitable linear layer and
    the provided activation function, which is expected to be a subclass of
    :class:`torch.nn.Module`, is added to the network. The linear layers are all either
    of type :class:`GUMLinear` or of type:class:`~torch.nn.Linear` based on the
    type of the expected input parameter for the activation function.

    Parameters
    ----------
    in_features : int
        number of channels of the input
    out_features : list[int]
        the hidden and output layers' dimensions
    activation_module : Type[Module]
        the activation function
    *args : int | float
        positional parameters to be forwarded to the constructor of the activation
        module
    **kwargs : int | float
        keyword parameters to be forwarded to the constructor of the activation module
    """

    def __init__(
        self,
        in_features: int,
        out_features: list[int],
        activation_module: Type[Module],
        *args: int | float,
        **kwargs: int | float,
    ) -> None:
        """An MLP consisting of stacked linear and provided activations"""
        self.activation_module = activation_module
        linear_module = (
            GUMLinear
            if signature(activation_module.forward).return_annotation is UncertainTensor
            else Linear
        )
        layers = ModuleList()
        for out_dimen in out_features:
            layers.append(linear_module(in_features, out_dimen))
            layers.append(activation_module(*args, **kwargs))
            in_features = out_dimen
        super().__init__(*layers)


class QuadLUMLP(MLP):
    """This implements the multi-layer perceptron (MLP) with QuadLU activation

    Parameters
    ----------
    in_features : int
        input dimension
    out_features : list[int]
        the hidden and output layers' dimensions
    """

    def __init__(self, in_features: int, out_features: list[int]):
        """An MLP consisting of linear layers and QuadLU activation function"""
        super().__init__(in_features, out_features, QuadLU)


class GUMQuadLUMLP(MLP):
    """This implements the multi-layer perceptron (MLP) with GUMQuadLU activation

    Parameters
    ----------
    in_features : int
        input dimension
    out_features : list[int]
        the hidden and output layers' dimensions
    """

    def __init__(self, in_features: int, out_features: list[int]) -> None:
        """An MLP consisting of GUMLinear and GUMQuadLU layers"""
        super().__init__(in_features, out_features, GUMQuadLU)


class GUMSoftplus(Module):
    r"""Implementation of parametrized Softplus activation with uncertainty propagation

    :math:`\operatorname{Softplus}_\beta \colon \mathbb{R} \to \mathbb{R}` is
    defined
    as

    .. math::

        \operatorname{Softplus}_\beta (x) := \frac{1}{\beta} \ln(1-\exp(\beta
        \cdot x))

    with :math:`\beta \in \mathbb{R}_+`. The uncertainty propagation is performed
    as stated in the thesis

    .. math::

        \mathbf{U}_{x^{(\ell)}} = \prod_{i=0}^{\ell-1} \operatorname{diag}
        \sigma(\beta z^{(\ell - i)} W^{(\ell-i)} \mathbf{U}_{x^{(0)}} \prod_{
        i=1}^{\ell} {W^{(i)}}^T \operatorname{diag} \sigma(\beta z^{(\ell - i))},

    Parameters
    ----------
    beta : int, optional
        non-negative parameter of Softplus activation function, defaults to 1
    threshold : int, optional
        The linearization threshold of the Softplus activation function for numerical
        stability, defaults to 20, for details see :class:`~torch.nn.Softplus`
    """

    def __init__(
        self,
        beta: int = 1,
        threshold: int = 20,
    ):
        """Parametrized Softplus activation function and uncertainty propagation"""
        super().__init__()
        self._softplus = Softplus(beta, threshold)

    def forward(self, uncertain_values: UncertainTensor) -> UncertainTensor:
        """Forward pass of GUMSoftplus"""
        with profiler.record_function("GUMSOFTPLUS PASS"):
            if uncertain_values.uncertainties is None:
                return UncertainTensor(
                    self._softplus(uncertain_values.values),
                    uncertain_values.uncertainties,
                )
            first_derivs = torch.sigmoid(self.beta * uncertain_values.values)
            return UncertainTensor(
                self._softplus(uncertain_values.values),
                first_derivs
                * uncertain_values.uncertainties
                * first_derivs.unsqueeze(1),
            )

    @property
    def beta(self) -> int:
        """The parameter beta of the activation function"""
        return self._softplus.beta

    @property
    def threshold(self) -> int:
        """The linearization threshold of the activation function"""
        return self._softplus.threshold


class GUMSoftplusMLP(MLP):
    """Implements the multi-layer perceptron (MLP) with GUMSoftplus activation

    Parameters
    ----------
    in_features : int
        input dimension
    out_features : list[int]
        the hidden and output layers' dimensions
    beta : int, optional
        non-negative parameter of Softplus activation function, defaults to 1
    threshold : int, optional
        The linearization threshold of the Softplus activation function for numerical
        stability, defaults to 20, for details see :class:`~torch.nn.Softplus`
    """

    def __init__(
        self,
        in_features: int,
        out_features: list[int],
        beta: int = 1,
        threshold: int = 20,
    ) -> None:
        """An MLP consisting of GUMLinear and GUMSoftplus layers"""
        super().__init__(in_features, out_features, GUMSoftplus, beta, threshold)


class GUMSigmoid(Module):
    r"""Implementation of Sigmoid activation with uncertainty propagation

    :math:`\sigma \colon \mathbb{R} \to \mathbb{R}` is
    defined
    as

    .. math::

        \sigma (x) := \frac{1}{1 + \exp(-x)}.

    The uncertainty propagation is performed as stated in the thesis

    .. math::

        \mathbf{U}_{x^{(\ell)}} = \prod_{i=0}^{\ell-1} \operatorname{diag}
        \sigma(z^{(\ell - i)}) \operatorname{diag} \big( \mathbf{1} - \sigma(z^{(\ell -
        i)}) \big) W^{(\ell-i)} \mathbf{U}_{x^{(0)}} \prod_{
        i=1}^{\ell} {W^{(i)}}^T \operatorname{diag}
        \sigma(z^{(\ell - i)} \operatorname{diag} \big( \mathbf{1} - \sigma(z^{(\ell -
        i)}) \big),
    """

    def __init__(self) -> None:
        """Sigmoid activation function with uncertainty propagation"""
        super().__init__()
        self._sigmoid = Sigmoid()

    def forward(self, uncertain_values: UncertainTensor) -> UncertainTensor:
        """Forward pass of GUMSigmoid"""
        with profiler.record_function("GUMSIGMOID PASS"):
            if uncertain_values.uncertainties is None:
                return UncertainTensor(self._sigmoid(uncertain_values.values), None)
            sigmoid_of_x = torch.sigmoid(uncertain_values.values)
            first_derivs = sigmoid_of_x * (1 - sigmoid_of_x)
            return UncertainTensor(
                self._sigmoid(uncertain_values.values),
                first_derivs
                * uncertain_values.uncertainties
                * first_derivs.unsqueeze(1),
            )


class GUMSigmoidMLP(MLP):
    """Implements the multi-layer perceptron (MLP) with GUMSigmoid activation

    Parameters
    ----------
    in_features : int
        input dimension
    out_features : list[int]
        the hidden and output layers' dimensions
    """

    def __init__(
        self,
        in_features: int,
        out_features: list[int],
    ):
        """An MLP consisting of GUMLinear and GUMSigmoid layers"""
        super().__init__(in_features, out_features, GUMSigmoid)
