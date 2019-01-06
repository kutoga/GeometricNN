from typing import List, Dict, Optional, Callable
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
        return reduce(lambda x_in, layer: layer.forward(x_in), self.__layers, x)

    def backward(self, prev: np.ndarray, derivative_rule: DerivativeRule) -> np.ndarray:
        return reduce(lambda prev_in, layer: layer.backward(prev_in, derivative_rule), reversed(self.__layers), prev)


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

def generate_exp_weight(shape: Optional[List[int]]) -> np.ndarray:
    return np.random.normal(size=shape)

WeightsGenerator = Callable[[Optional[List[int]]], np.ndarray]

class LinearLayer(Layer):
    def __init__(self, n_inputs: int, n_outputs: int, bias=True,
                 weights_generator: WeightsGenerator=np.random.normal,
                 bias_generator: WeightsGenerator=np.random.normal) -> None:
        self.__n_inputs = n_inputs
        self.__n_outputs = n_outputs
        self.__weights = weights_generator([self.__n_inputs, self.__n_outputs])
        self.__bias = bias_generator([self.__n_outputs]) if bias else None
        self.__state = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert len(x.shape) <= 2
        assert x.shape[-1] == self.__n_inputs
        if len(x.shape) == 1:
            y = x * self.__weights
            if self.__bias is not None:
                y += self.__bias
        elif len(x.shape) == 2:
            y = np.dot(x, self.__weights)
            if self.__bias is not None:
                y += self.__bias
        else:
            raise RuntimeError
        self.__state['x'] = x
        self.__state['y'] = y
        return y

    def backward(self, prev: np.ndarray, derivative_rule: DerivativeRule) -> np.ndarray:
        pass

