from typing import List, Callable
import pytest
import numpy as np

from sympy import exp

from mult_nn._math import MExpression, sym, maximum
from mult_nn._nn_ops import DerivativeRule
from mult_nn._nn_update import UpdateRule, gradient_descent, mul_gradient_descent
from mult_nn._nn import Layer, ActivationLayer, WeightsGenerator, LinearLayer, LayerCollection

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
    #([
    #    ActivationLayer(MExpression(x ** 2 + x)),
    #    ActivationLayer(MExpression(-x)),
    #    ActivationLayer(MExpression(exp(x)))
    # ], np.array([0., 1.]), np.array([1., 1.]), DerivativeRule.dx_rule(), np.array([-1., -3. * np.exp(-2.)])),
    ([
        ActivationLayer(MExpression(x ** 2 + 1)),
        ActivationLayer(MExpression(exp(x))),
        ActivationLayer(MExpression(x * 13.)),
        ActivationLayer(MExpression(x + 1))
     ], np.array([0., 1.]), np.array([np.exp(1.), np.exp(1.)]), DerivativeRule.ux_rule(), np.array([1., np.exp((26. * np.exp(2.)) / (13 * np.exp(2.) + 1))]))
])
def test_layer_collection_activation_backward(layers: List[Layer], input: np.ndarray, previous: np.ndarray,
                                              derivative_rule: DerivativeRule, expected_output: np.ndarray) -> None:
    lc = LayerCollection(*layers)
    lc.forward(input)
    backward_output = lc.backward(previous, derivative_rule)

    np.testing.assert_almost_equal(backward_output, expected_output)

@pytest.mark.parametrize('layer_builder, weights, bias, input, expected_output', [
     ((lambda weights_gen, bias_gen: LinearLayer(3, 2, True, weights_gen, bias_gen)),
      np.array([[1., 3.], [2., 4.], [-1., -2.]]), np.array([5., 6.]),
      np.array([[2., 3., -1.]]), np.array([[14., 26.]])),
    ((lambda weights_gen, bias_gen: LinearLayer(2, 1, True, weights_gen, bias_gen)),
     np.array([[1.], [2.]]), np.array([-1.]),
     np.array([[4., 5.], [2., 3.]]), np.array([[13.], [7.]]))
])
def test_linear_activation(layer_builder: Callable[[WeightsGenerator, WeightsGenerator], LinearLayer],
                           weights: np.ndarray, bias: np.ndarray, input: np.ndarray,
                           expected_output: np.ndarray) -> None:
    layer = layer_builder(lambda _: weights, lambda _: bias)
    output = layer.forward(input)

    np.testing.assert_almost_equal(output, expected_output)

@pytest.mark.parametrize('update_rule, weights, gradient, expected_updated_weights', [
    (gradient_descent(), np.array([1., 2., 3.]), np.array([2., 1., -2.]), np.array([-1., 1., 5.])),
    (mul_gradient_descent(), np.array([1., 2., 3.]), np.array([0.5, 0.1, 5.]), np.array([2., 20., 0.6]))
])
def test_update_rule_should_update_weights(update_rule: UpdateRule, weights: np.ndarray, gradient: np.ndarray,
                     expected_updated_weights: np.ndarray) -> None:
    updated_weights = update_rule.update(weights, gradient)

    np.testing.assert_almost_equal(updated_weights, expected_updated_weights)

@pytest.mark.parametrize('weights', [
    np.array([1., 2., -1.]),
    np.array([0., 1., 3.])
])
def test_mul_gradient_descent_raises_on_non_positive_weights(weights: np.ndarray) -> None:
    update_rule = mul_gradient_descent()
    dummy_gradient = np.ones_like(weights)

    with pytest.raises(ValueError):
        update_rule.update(weights, dummy_gradient)

@pytest.mark.parametrize('gradient', [
    np.array([3., -2., 0.5]),
    np.array([1., 0., 10.])
])
def test_mul_gradient_descent_raises_on_non_positive_gradient(gradient: np.ndarray) -> None:
    update_rule = mul_gradient_descent()
    dummy_weights = np.ones_like(gradient)

    with pytest.raises(ValueError):
        update_rule.update(dummy_weights, gradient)
#
# def test_linear_activation_backprop(layer_builder: Callable[[WeightsGenerator, WeightsGenerator], LinearLayer],
#                                     weights: np.ndarray, bias: np.ndarray, input: np.ndarray,
#                                     previous: np.ndarray, derivative_rule: DerivativeRule,
#                                     expected_output: np.ndarray, exptected_weights_gradient: np.ndarray,
#                                     expected_bias_gradient: np.ndarray) -> None:
#     layer = layer_builder(lambda _: weights, lambda _: bias)
#     layer.forward(input)
#     backward_output = layer.backward(previous, derivative_rule)
#     layer.update_weights(update_rule=lambda gradient, weights: weights - gradient)
#
#     assert backward_output == expected_output