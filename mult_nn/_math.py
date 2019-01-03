from typing import Callable, Tuple, List, Any, Union
from functools import lru_cache
from sympy import symbols, diff, exp, lambdify, Symbol, Expr, sympify

from ._utils import multiple_input_params

MSymbol = Symbol


@multiple_input_params
def sym(name: str) -> MSymbol:
    return symbols(name)

_x = sym('x')

class MExpression:
    def __init__(self, sympy_expr: Union[Expr, float]):
        self.__sympy_expr = sympy_expr if not isinstance(sympy_expr, float) else sympify(sympy_expr)
        self.__lambdify = lru_cache(maxsize=256)(self.__lambdify)

    def __repr__(self) -> str:
        return f'MExpression({repr(self.__sympy_expr)})'

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, MExpression) and other.__sympy_expr == self.__sympy_expr

    def __hash__(self) -> int:
        return hash(self.__sympy_expr)

    def __call__(self, *symbol_values: Tuple[MSymbol, Any], **variables):
        used_symbols: List[Tuple[str, Any]] = sorted(variables.items())
        used_symbols.extend([(symbol.name, value) for symbol, value in symbol_values])
        f = self.__lambdify(tuple(name for name, _ in used_symbols))
        return MExpression(f(*(value for _, value in used_symbols)))

    def __lambdify(self, used_symbols: Tuple[str]) -> Callable:
        return lambdify(list(map(symbols, used_symbols)), self.__sympy_expr, modules=['numpy'])

    def sympy(self) -> Expr:
        return self.__sympy_expr

    def sympy_apply(self, operator: Callable[[Expr], Expr]) -> 'MExpression':
        return MExpression(operator(self.__sympy_expr))


def dydx(y: MExpression, x: MSymbol=_x) -> MExpression:
    return y.sympy_apply(lambda expr: diff(expr, x))

def uyux(y: MExpression, x: MSymbol=_x) -> MExpression:
    return y.sympy_apply(lambda expr: exp(diff(expr, x) / expr))

