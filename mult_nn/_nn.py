from typing import List, Dict, Optional, Callable
from abc import abstractmethod, ABC
from functools import reduce

import numpy as np

from ._math import MExpression, MSymbol, sym
from ._nn_ops import DerivativeRule
from ._nn_update import UpdateRule

_x = sym('x')

class Layer(ABC):

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def backward(self, prev: np.ndarray, derivative_rule: DerivativeRule) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def update_weights(self, update_rule: UpdateRule) -> None:
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

    def update_weights(self, update_rule: UpdateRule):
        for layer in self.__layers:
            layer.update_weights(update_rule)


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

    def update_weights(self, update_rule: UpdateRule):
        pass


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
        assert self.__weights.shape == (self.__n_outputs, self.__n_inputs)
        self.__bias = bias_generator([self.__n_outputs]) if bias else None
        if self.__bias is not None:
            assert self.__bias.shape == (self.__n_outputs,)
        self.__state = {}

    @property
    def weights(self) -> np.ndarray:
        return self.__weights

    @property
    def bias(self) -> Optional[np.ndarray]:
        return self.__bias

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert len(x.shape) <= 2
        assert x.shape[-1] == self.__n_inputs
        if len(x.shape) == 1:
            x = np.array([x], dtype=x.dtype)
        if len(x.shape) == 2:
            y = np.dot(x, np.transpose(self.__weights))
            if self.__bias is not None:
                y += self.__bias
        else:
            raise RuntimeError
        self.__state['x'] = x
        self.__state['y'] = y
        return y

    def backward(self, prev: np.ndarray, derivative_rule: DerivativeRule) -> np.ndarray:
        if derivative_rule is not DerivativeRule.dx_rule():
            raise ValueError('The linear layer currently only supports dydx rule.')
        self.__state['derivative_x'] = np.dot(prev, self.__weights)
        self.__state['derivative_w'] = np.dot(np.expand_dims(prev, -1), np.expand_dims(self.__state['x'], 0))[:, 0, :]
        self.__state['derivative_b'] = prev
        return self.__state['derivative_x']

    def update_weights(self, update_rule: UpdateRule):
        self.__weights = update_rule.update(self.__weights, self.__state['derivative_w'])
        if self.__bias is not None:
            self.__bias = update_rule.update(self.__bias, self.__state['derivative_b'])
