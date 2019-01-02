from typing import List
from abc import abstractmethod, ABC
from functools import reduce

import numpy as np

from ._math import MExpression


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
