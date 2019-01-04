from typing import Callable
from functools import lru_cache

import numpy as np

from ._math import MExpression, MSymbol, dydx, uyux

DerivativeOperator = Callable[[MExpression, MSymbol], MExpression]
ChainRuleOperator = Callable[[np.ndarray, np.ndarray, MExpression, MSymbol], np.ndarray]

class DerivativeRule:
    def __init__(self, derivative_op: DerivativeOperator, chain_rule_op: ChainRuleOperator) -> None:
        self.__derivative_op = derivative_op
        self.__chain_rule_op = chain_rule_op

    def derivative(self, f: MExpression, x: MSymbol) -> MExpression:
        return self.__derivative_op(f, x)

    def chain_rule(self, derivative_g_x: np.ndarray, g_x: np.ndarray, f: MExpression, x: MSymbol) -> np.ndarray:
        return self.__chain_rule_op(derivative_g_x, g_x, f, x)

    @staticmethod
    @lru_cache(maxsize=1)
    def dx_rule() -> 'DerivativeRule':
        def dx_chain(derivative_g_x: np.ndarray, g_x: np.ndarray, f: MExpression, x: MSymbol) -> np.ndarray:
            return dydx(f, x)((x, g_x)).sympy() * derivative_g_x
        return DerivativeRule(dydx, dx_chain)

    @staticmethod
    @lru_cache(maxsize=1)
    def ux_rule() -> 'DerivativeRule':
        def ux_chain(derivative_g_x: np.ndarray, g_x: np.ndarray, f: MExpression, x: MSymbol) -> np.ndarray:
            return uyux(f, x)((x, g_x)).sympy() ** (g_x * np.log(derivative_g_x))
        return DerivativeRule(uyux, ux_chain)



