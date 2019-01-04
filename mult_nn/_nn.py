from typing import List, Dict
from abc import abstractmethod, ABC
from functools import reduce

import numpy as np

from ._math import MExpression, MSymbol, sym
from ._nn_ops import DerivativeRule

_x = sym('x')

class Layer(ABC):

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def backward(self, prev: np.ndarray, derivative_rule: DerivativeRule) -> np.ndarray:
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

    def backward(self, prev: np.ndarray, derivative_rule: DerivativeRule) -> np.ndarray:
        return reduce(lambda layer, prev_in: layer.backward(prev_in, derivative_rule), reversed(self.__layers), prev)


class ActivationLayer(Layer):
    def __init__(self, func: MExpression) -> None:
        self.__func = func
        self.__state: Dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = self.__func(x=x).sympy()
        self.__state = {
            'x': x,
            'y': y
        }
        return y

    def backward(self, prev: np.ndarray, derivative_rule: DerivativeRule) -> np.ndarray:
        x = self.__state['x']
        y = self.__state['y']
        derivative_x = derivative_rule.chain_rule(prev, x, self.__func, _x)
        self.__state['derivative_x'] = derivative_x
        return derivative_x