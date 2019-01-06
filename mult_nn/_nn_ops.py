from typing import Callable

import numpy as np

from sympy import Expr

from ._math import MExpression, MSymbol, dydx, uyux

DerivativeOperator = Callable[[MExpression, MSymbol], MExpression]
ChainRuleOperator = Callable[[np.ndarray, np.ndarray, MExpression, MSymbol], np.ndarray]

class DerivativeRule:
    __DX_RULE = None
    __UX_RULE = None

    def __init__(self, derivative_op: DerivativeOperator, chain_rule_op: ChainRuleOperator) -> None:
        self.__derivative_op = derivative_op
        self.__chain_rule_op = chain_rule_op

    def derivative(self, f: MExpression, x: MSymbol) -> MExpression:
        return self.__derivative_op(f, x)

    def chain_rule(self, derivative_g_x: np.ndarray, g_x: np.ndarray, f: MExpression, x: MSymbol) -> np.ndarray:
        return self.__chain_rule_op(derivative_g_x, g_x, f, x)

    @staticmethod
    def _try_numpify(x: Expr) -> Expr:
        if isinstance(x, np.ndarray):
            return x.astype(float)
        return x

    @staticmethod
    def __dx_rule() -> 'DerivativeRule':
        def dx_chain(derivative_g_x: np.ndarray, g_x: np.ndarray, f: MExpression, x: MSymbol) -> np.ndarray:
            return DerivativeRule._try_numpify(dydx(f, x)((x, g_x)).sympy() * derivative_g_x)
        return DerivativeRule(dydx, dx_chain)

    @staticmethod
    def __ux_rule() -> 'DerivativeRule':
        def ux_chain(derivative_g_x: np.ndarray, g_x: np.ndarray, f: MExpression, x: MSymbol) -> np.ndarray:
            return DerivativeRule._try_numpify(uyux(f, x)((x, g_x)).sympy() ** (g_x * np.log(derivative_g_x)))
        return DerivativeRule(uyux, ux_chain)

    @staticmethod
    def dx_rule() -> 'DerivativeRule':
        if DerivativeRule.__DX_RULE is None:
            DerivativeRule.__DX_RULE = DerivativeRule.__dx_rule()
        return DerivativeRule.__DX_RULE

    @staticmethod
    def ux_rule() -> 'DerivativeRule':
        if DerivativeRule.__UX_RULE is None:
            DerivativeRule.__UX_RULE = DerivativeRule.__ux_rule()
        return DerivativeRule.__UX_RULE


