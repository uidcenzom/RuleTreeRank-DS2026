from collections import namedtuple
from typing import runtime_checkable, Protocol, Union, Dict

from numpy import ndarray
from pandas import DataFrame, Series

ModelParam = namedtuple('ModelParam', ['model', 'param'])


@runtime_checkable
class TreeModel(Protocol):

    def fit(self, X: Union[DataFrame, ndarray], y: Union[Series, ndarray]) -> "TreeModel": ...

    def predict(self, X: Union[DataFrame, ndarray]) -> ndarray: ...

    def apply(self, X: Union[DataFrame, ndarray]) -> ndarray: ...

    def get_rules(self) -> Dict: ...

@runtime_checkable
class RankerModel(Protocol):

    def __init__(self, *args, **kwargs): ...

    def fit(self, X: ndarray, y: ndarray, q: ndarray, *args, **kwargs) -> 'RankerModel': ...

    def predict(self, X: ndarray, q: ndarray, *args, **kwargs) -> ndarray: ...

__all__ = [
    "ModelParam",
    "TreeModel",
    "RankerModel",
]
