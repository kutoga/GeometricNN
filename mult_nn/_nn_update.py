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

_gradient_descent = GradientDescent()

def gradient_descent() -> GradientDescent:
    return _gradient_descent


class MulGradientDescent(UpdateRule):
    def update(self, weights: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        if (np.sum(weights <= 0.) > 0) or (np.sum(gradient <= 0.) > 0):
            raise ValueError
        return weights / gradient

_mul_gradient_descent = MulGradientDescent()

def mul_gradient_descent() -> MulGradientDescent:
    return _mul_gradient_descent
