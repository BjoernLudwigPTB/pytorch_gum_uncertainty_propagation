from torch.nn import Module

from GUM_compliant_neural_network_uncertainty_propagation.activations import QuadLU


def test_init_quadlu():
    assert QuadLU()


def test_quadlu_is_subclass_of_nn_module():
    assert issubclass(QuadLU, Module)
