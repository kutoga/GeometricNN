import pytest
import numpy as np

from sympy import Max

from mult_nn._math import MExpression, sym, maximum
from mult_nn._nn_ops import DerivativeRule
from mult_nn._nn import Layer, ActivationLayer

x = sym('x')

@pytest.mark.parametrize('layer, input, expected_output', [
    (ActivationLayer(MExpression(2 ** x)), np.array([1., 2., 0.5]), np.array([2., 4., np.sqrt(2)]))
])
def test_forward_activation(layer: Layer, input: np.ndarray, expected_output: np.ndarray) -> None:
    output = layer.forward(input)

    np.testing.assert_almost_equal(output, expected_output)

@pytest.mark.parametrize('layer, input, previous, derivative_rule, expected_output', [
    (ActivationLayer(MExpression(x ** 2)),
     np.array([1., 2.]), np.array([3., 4.]),
     DerivativeRule.dx_rule(), np.array([6., 16.])),

    (ActivationLayer(MExpression(maximum(x, 0))),
     np.array([1., -1., 2.]), np.array([-2., 2., -1.]),
     DerivativeRule.dx_rule(), np.array([-2., 0., -1.]))
])
def test_backward_activation(layer: Layer, input: np.ndarray, previous: np.ndarray, derivative_rule: DerivativeRule,
                             expected_output: np.ndarray) -> None:
    layer.forward(input)
    backward_output = layer.backward(previous, derivative_rule)

    np.testing.assert_almost_equal(backward_output, expected_output)
