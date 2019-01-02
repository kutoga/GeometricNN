from typing import Dict

import pytest
from sympy import Expr, sympify, exp
from mult_nn._math import MExpression, sym, dydx, uyux

x, y, z = sym('x', 'y', 'z')

@pytest.mark.parametrize('expression, parameters, expected', [
    (x + 1, {'x':1.}, sympify(2)),
    (x + y * z, {'x':2., 'y':3.}, 2 + 3 * z)
])
def test_call(expression: Expr, parameters: Dict[str, float], expected: Expr) -> None:
    assert MExpression(expression)(**parameters) == MExpression(expected)

@pytest.mark.parametrize('expression, derivative', [
    (x + 1, 1),
    (exp(x + y), exp(x + y)),
    (x**2*y, 2*x*y)
])
def test_dydx(expression: Expr, derivative: Expr) -> None:
    assert dydx(MExpression(expression), x) == MExpression(derivative)

@pytest.mark.parametrize('expression, multiplicative_derivative', [
    (2**x, 2),
    (exp(exp(x))*y, exp(exp(x)))
])
def test_uyux(expression: Expr, multiplicative_derivative: Expr) -> None:
    assert uyux(MExpression(expression), x) == MExpression(multiplicative_derivative)
