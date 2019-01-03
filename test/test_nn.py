import pytest
import numpy as np

from mult_nn._math import MExpression, sym
from mult_nn._nn import Layer, ActivationLayer

x = sym('x')

@pytest.mark.parametrize('layer, input, expected_output', [
    (ActivationLayer(MExpression(2 ** x)), np.array([1., 2., 0.5]), np.array([2., 4., np.sqrt(2)]))
])
def test_forward_activation(layer: Layer, input: np.ndarray, expected_output: np.ndarray) -> None:
    output = layer.forward(input)

    np.testing.assert_almost_equal(output, expected_output)

def test_backward_activation(layer: Layer, previous: np.ndarray, expected_output: np.ndarray) -> None:
    assert False


