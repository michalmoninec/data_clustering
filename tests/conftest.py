import numpy as np

from pytest import fixture, FixtureRequest
from typing import Union, Type

from src.clustering.utils.algorithms import KMeans, DBSCAN, MeanShift
from src.clustering.utils.data_model import (
    KMeansModel,
    DBSCANModel,
    MeanShiftModel,
)


AlgoType = Union[Type[KMeans], Type[DBSCAN], Type[MeanShift]]
ModelType = Union[Type[KMeansModel], Type[DBSCANModel], Type[MeanShiftModel]]


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


@fixture
def attr_name(request: FixtureRequest) -> str:
    """Fixture that returns the attribute name based on the provided parameter.

    Args:
        request (FixtureRequest): The request object.

    Returns:
        str: The attribute name.
    """
    return request.param


@fixture
def attr_class(request: FixtureRequest) -> Type[object]:
    """Fixture that returns a specific class, based on provided parameter.

    Args:
        request (FixtureRequest): The request object.

    Returns:
        Type[object]: The class object.
    """
    return request.param


@fixture
def model_class(request: FixtureRequest) -> ModelType:
    """Fixture, that returns ModelType for provided class."""
    return request.param
