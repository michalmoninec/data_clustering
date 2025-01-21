import numpy as np

from pytest import fixture, FixtureRequest
from typing import Union, Type

from src.clustering.utils.algorithms import KMeans

AlgoType = Union[Type[KMeans]]


@fixture
def mock_data() -> np.ndarray:
    """Fixture that creates a mock data object.

    Returns:
        np.ndarray: A mock data object.
    """
    return np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])


@fixture
def algo_class(request: FixtureRequest) -> AlgoType:
    """Fixture that returns a specific class, based on provided parameter.

    Args:
        request (FixtureRequest): The request object.

    Returns:
        AlgoType: The class object.
    """
    return request.param
