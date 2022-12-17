"""Test the module modules"""
from pytorch_gum_uncertainty_propagation import modules


def test_modules_has_module_docstring() -> None:
    assert modules.__doc__ is not None
