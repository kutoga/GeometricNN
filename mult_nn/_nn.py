from typing import List, Callable, NamedTuple
from abc import abstractmethod, ABC
from functools import reduce

import numpy as np

from ._math import MExpression, MSymbol

DerivativeOperator = Callable[[MExpression, MSymbol], MExpression]
ChainRuleOperator = Callable[[MExpression, ...], MExpression]

class DerivativeRule:
    def __init__(self, derivative_op: DerivativeOperator, chain_rule_op: ChainRuleOperator) -> None:
        self.__derivative_op = derivative_op
        self.__chain_rule_op = chain_rule_op

    def derivative(self, f: MExpression, x: MSymbol) -> MExpression:
        pass

    def chain_rule(self, derivative_g_x: float, g_x: float, f: MExpression, x: MSymbol) -> MExpression:
        pass

    @staticmethod
    def from_mexp_op(derivative: DerivativeOperator) -> 'DerivativeRule':
        return DerivativeRule(derivative, )
        pass


class Layer(ABC):

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def backward(self, prev: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class LayerCollection(Layer):
    def __init__(self, *layers: Layer) -> None:
        self.__layers = list(layers)

    def __repr__(self):
        return f'LayerCollection[{repr(self.__layers)}]'

    @property
    def layers(self) -> List[Layer]:
        return self.__layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        return reduce(lambda layer, x_in: layer.forward(x_in), self.__layers, x)

    def backward(self, prev: np.ndarray) -> np.ndarray:
        return reduce(lambda layer, prev_in: layer.backward(prev_in), reversed(self.__layers), prev)


class ActivationLayer(Layer):
    def __init__(self, func: MExpression) -> None:
        self.__func = func

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.__func(x=x).sympy()

    def backward(self, prev: np.ndarray) -> None:
        pass
