from typing import List
import pytest
import numpy as np

from sympy import exp

from mult_nn._math import MExpression, sym, maximum
from mult_nn._nn_ops import DerivativeRule
from mult_nn._nn import Layer, ActivationLayer, LinearLayer, LayerCollection

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

@pytest.mark.parametrize('layers, input, expected_output', [
    ([
        ActivationLayer(MExpression(x ** 2. + x)),
        ActivationLayer(MExpression(exp(-x)))
     ], np.array([2., 3.]), np.array([np.exp(-6.), np.exp(-12.)])),
    ([
        ActivationLayer(MExpression(2. ** x * 1.5 ** x)),
        ActivationLayer(MExpression(maximum(x, 2.)))
     ], np.array([[1.], [.5]]), np.array([[3.], [2.]]))
])
def test_layer_collection_activation_forward(layers: List[Layer], input: np.ndarray, expected_output: np.ndarray) -> None:
    lc = LayerCollection(*layers)
    output = lc.forward(input)

    np.testing.assert_almost_equal(output, expected_output)

@pytest.mark.parametrize('layers, input, previous, derivative_rule, expected_output', [
    ([
        ActivationLayer(MExpression(x ** 2 + x)),
        ActivationLayer(MExpression(-x)),
        ActivationLayer(MExpression(exp(x)))
     ], np.array([0., 1.]), np.array([1., 1.]), DerivativeRule.dx_rule(), np.array([-1., -3. * np.exp(-2.)])),
    ([
        ActivationLayer(MExpression(x ** 2)),
        ActivationLayer(MExpression(exp(x))),
        ActivationLayer(MExpression(x * 13.)),
        ActivationLayer(MExpression(x + 1))
     ], np.array([0., 1.]), np.array([1., 1.]), DerivativeRule.ux_rule(), np.array([1., np.exp((26. * np.exp(1.)) / (13 * np.exp(1.) + 1))]))
])
def test_layer_collection_activation_backward(layers: List[Layer], input: np.ndarray, previous: np.ndarray,
                                              derivative_rule: DerivativeRule, expected_output: np.ndarray) -> None:
    lc = LayerCollection(*layers)
    lc.forward(input)
    backward_output = lc.backward(previous, derivative_rule)

    np.testing.assert_almost_equal(backward_output, expected_output)

# @pytest.mark.parametrize('layer, input, expected_output', [
#     (LinearLayer(3, 2, bias=True, weights=np.array([[1., 2., -1.], [3., 4., -2.]]), bias_weights=np.array([5., 6.])),
#      np.array([[2., 3., -1.]]), np.array([[14., 26.]]))
# ])
# def test_linear_activation(layer: LinearLayer, input: np.ndarray, expected_output: np.ndarray) -> None:
#     output = layer.forward(input)
#
#     np.testing.assert_almost_equal(output, expected_output)
