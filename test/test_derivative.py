import pytest

import numpy as np

from mult_nn._math import dydx, uyux, sym, MExpression
from mult_nn._nn_ops import DerivativeRule

x = sym('x')

def test_identity_derivative_rule_derivative_should_be_identity() -> None:
    f = MExpression(2 ** x)
    identity_derivative_rule = DerivativeRule(lambda mexpr, _: mexpr, lambda _, _1, mexpr, _2: mexpr)

    assert identity_derivative_rule.derivative(f, x) == f

def test_identity_derivative_rule_chain_rule_should_return_identity() -> None:
    f = MExpression(2 ** x)
    derivative_g_x = 3
    g_x = 3
    identity_derivative_rule = DerivativeRule(lambda mexpr, _: mexpr, lambda _, g_x, mexpr, _1: mexpr((x, g_x)).sympy())

    np.testing.assert_almost_equal(identity_derivative_rule.chain_rule(derivative_g_x, g_x, f, x), 8)

@pytest.mark.parametrize('f, x0, expected_derivative', [
    (MExpression(x ** 2), 1., 2.),
    (MExpression(x ** 2 + x ** 3), 2., 16.)
])
def test_dx_derivative_rule_derivative_should_return(f: MExpression, x0: float, expected_derivative: float) -> None:
    d_rule = DerivativeRule.dx_rule()

    derivative = d_rule.derivative(f, x)((x, x0))

    np.testing.assert_almost_equal(derivative.sympy(), expected_derivative)

@pytest.mark.parametrize('f, derivative_g_x, g_x, expected_result', [
    (MExpression(2 ** x), 2., 3., 2 ** (3. * np.log(2.))),
    (MExpression(x ** 2), 3., 4., np.exp(2. / 4.) ** (4. * np.log(3.)))
])
def test_ux_derivative_rule_chain_rule_should_return(f: MExpression, derivative_g_x: float, g_x: float, expected_result: float) -> None:
    d_rule = DerivativeRule.ux_rule()

    chain_rule_result = d_rule.chain_rule(derivative_g_x, g_x, f, x)

    np.testing.assert_almost_equal(chain_rule_result, expected_result)
