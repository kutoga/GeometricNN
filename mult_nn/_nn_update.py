from typing import Callable
from abc import ABC, abstractmethod

import numpy as np

class UpdateRule(ABC):

    @abstractmethod
    def update(self, weights: np.ndarray, gradient: np.ndarray):
        raise NotImplementedError

class GradientDescent(UpdateRule):
    def update(self, weights: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        return weights - gradient


class MulGradientDescent(UpdateRule):
    def update(self, weights: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        if (np.sum(weights <= 0.) > 0) or (np.sum(gradient <= 0.) > 0):
            raise ValueError
        return weights / gradient

class Mul2AddGradientDescent(UpdateRule):
    def __init__(self):
        self.__gradient_descent = GradientDescent()
        self.__y = None

    def update_y(self, y: float) -> None:
        self.__y = y

    def update(self, weights: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        if np.sum(gradient <= 0.) > 0:
            raise ValueError
        if self.__y is None:
            raise ValueError('y is not defined')
        additive_gradient = self.__y * np.log(gradient)
        return self.__gradient_descent.update(weights, additive_gradient)


