"""Fixture for all tests related to the module 'modules'"""
import pytest

from gum_compliant_neural_network_uncertainty_propagation.modules import (
    QuadLU,
    UncertainQuadLU,
)


@pytest.fixture()
def quadlu() -> QuadLU:
    return QuadLU()


@pytest.fixture()
def uncertain_quadlu() -> UncertainQuadLU:
    return UncertainQuadLU()
