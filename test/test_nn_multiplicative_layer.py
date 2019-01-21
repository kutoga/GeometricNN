import pytest
import numpy as np

from mult_nn._nn_ops import DerivativeRule
from mult_nn._nn_update import MulGradientDescent
from mult_nn._nn import MultiplicativeLayer


@pytest.mark.parametrize('weights, bias, x, expected_y', [
    (np.array([[1., 2., 3.]]), np.array([1.]), np.array([2., 1., 2.]), np.array([[18.]])),
    (np.array([[1., 2.], [3., 1.], [1., 2.]]), np.array([1., 1., 2.]), np.array([3., 1.]), np.array([[2., 27., 4.]])),
])
def test_multiplicative_activation_forward_pass(weights: np.ndarray, bias: np.ndarray, x: np.ndarray, expected_y: np.ndarray) -> None:
    layer = MultiplicativeLayer(weights.shape[1], weights.shape[0], weights_generator=lambda _: weights, bias_generator=lambda _: bias)

    y = layer.forward(x)

    np.testing.assert_almost_equal(y, expected_y)

@pytest.mark.parametrize('weights, bias', [
    (np.array([[-1., 0.]]), None),
    (np.array([[11., 1.]]), np.array([-1.])),
    (np.array([[0., 1.]]), np.array([1.])),
    (np.array([[2., 1.]]), np.array([0.])),
])
def test_non_positive_weights_should_not_be_allowed(weights: np.ndarray, bias: np.ndarray) -> None:
    with pytest.raises(AssertionError):
        MultiplicativeLayer(weights.shape[1], weights.shape[0], weights_generator=lambda _: weights,
                            bias=bias is not None, bias_generator=lambda _: bias)

@pytest.mark.parametrize('bias', [
    np.array([1.]), np.array([-4.]), np.array([0.]), None
])
@pytest.mark.parametrize('weights, input, previous, expected_output', [
    #(np.array([[1., -1., 2.]]), np.array([2., -1., 3.]), np.array([1.]), np.array([1., -1., 2.])),
    #(np.array([[0., -3., 22.]]), np.array([12., -1111., -13.]), np.array([-1.]), np.array([0., 3., -22.])),
])
def test_multiplicative_activation_backprop_with_single_output(weights: np.ndarray, bias: np.ndarray, input: np.ndarray,
                                    previous: np.ndarray, expected_output: np.ndarray) -> None:
    derivative_rule = DerivativeRule.ux_rule()
    layer = MultiplicativeLayer(weights.shape[1], weights.shape[0], weights_generator=lambda _: weights, bias_generator=lambda _: bias)
    layer.forward(input)

    backward_output = layer.backward(previous, derivative_rule)

    np.testing.assert_almost_equal(backward_output, expected_output)

"""
@pytest.mark.parametrize('weights, bias, input, previous, expected_output', [
    (np.array([[2., 3., -1.], [0., -0.5, 1.]]), np.array([1., 2.]), np.array([1., 2., -1.]),
     np.array([1., 1.]), np.array([2., 2.5, 0.])),
    (np.array([[2., 3., -1.], [0., -0.5, 1.]]), np.array([1., 2.]), np.array([1., 2., -1.]),
     np.array([1., 2.]), np.array([2., 2., 1.]))
])
def test_multiplicative_activation_backprop_with_multiple_outputs(weights: np.ndarray, bias: np.ndarray, input: np.ndarray,
                                                          previous: np.ndarray, expected_output: np.ndarray) -> None:
    derivative_rule = DerivativeRule.ux_rule()
    layer = MultiplicativeLayer(weights.shape[1], weights.shape[0], weights_generator=lambda _: weights, bias_generator=lambda _: bias)
    layer.forward(input)

    backward_output = layer.backward(previous, derivative_rule)

    np.testing.assert_almost_equal(backward_output, expected_output)

@pytest.mark.parametrize('weights, bias, input, previous, expected_updated_weights, expected_updated_bias', [
    (np.array([[1., 2.], [3., 4.]]), np.array([-1., 2.]), np.array([3., 4.]), np.array([2., -2.]),
     np.array([[-5, -6.], [9., 12.]]), np.array([-3., 4.]))
])
def test_multiplicative_activation_backprop_weights(weights: np.ndarray, bias: np.ndarray, input: np.ndarray,
                                            previous: np.ndarray, expected_updated_weights: np.ndarray,
                                            expected_updated_bias: np.ndarray) -> None:
    derivative_rule = DerivativeRule.ux_rule()
    update_rule = MulGradientDescent()
    layer = MultiplicativeLayer(weights.shape[1], weights.shape[0], weights_generator=lambda _: weights, bias_generator=lambda _: bias)

    layer.forward(input)
    layer.backward(previous, derivative_rule)
    layer.update_weights(update_rule)

    np.testing.assert_almost_equal(layer.weights, expected_updated_weights)
    np.testing.assert_almost_equal(layer.bias, expected_updated_bias)


def test_multiplicative_activation_backprop_without_bias() -> None:
    layer = MultiplicativeLayer(5, 5, bias=False, weights_generator=lambda shape: np.zeros(shape, dtype=np.float32))
    derivative_rule = DerivativeRule.ux_rule()

    layer.forward(np.zeros((5,), dtype=np.float32))
    layer.backward(np.zeros((5,), dtype=np.float32), derivative_rule)

    assert layer.weights is not None
    assert layer.bias is None
"""
