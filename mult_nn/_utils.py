from typing import Callable, Union, List, TypeVar
from functools import wraps

_U = TypeVar('_U')
_V = TypeVar('_V')


def multiple_input_params(func: Callable[[_U], _V]) -> Callable[..., Union[_V, List[_V]]]:
    @wraps(func)
    def _wrapped_func(*args: _U, always_return_tuple: bool=False) -> Union[_V, List[_V]]:
        results = tuple(map(func, args))
        if len(results) == 1 and not always_return_tuple:
            return results[0]
        return results
    return _wrapped_func

