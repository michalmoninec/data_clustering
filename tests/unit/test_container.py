import pytest

from typing import Type, Union
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.cluster import MeanShift as SklearnMeanShift
from unittest.mock import Mock

from src.clustering.utils.container import Container
from src.clustering.utils.algorithms import KMeans, DBSCAN, MeanShift

AlgoSklearnType = Union[
    Type[SklearnKMeans], Type[SklearnDBSCAN], Type[SklearnMeanShift]
]
AlgoType = Union[Type[KMeans], Type[DBSCAN], Type[MeanShift]]


@pytest.mark.parametrize(
    "mock_dict_config",
    [
        ("sklearn_kmeans_dict_config"),
        ("sklearn_dbscan_dict_config"),
        ("sklearn_mean_shift_dict_config"),
    ],
    indirect=True,
)
def test_container_config_init(mock_dict_config: dict) -> None:
    """Test the initialization of Container and initialization of config
    instance with mocked configuration dictionary.

    Args:
        mock_dict_config (dict): Mock configuration for corresponding
        sklearn class.
        attr_name (str): Container attribute name for corresponding sklearn
        class.

    Asserts:
        The container is initialized correctly without raising an error.
        The instance of container config is initialized correctly.
    """
    container = Container()
    container.config.from_dict(mock_dict_config)
    container.config()


@pytest.mark.parametrize(
    "mock_dict_config, attr_name, algo_class",
    [
        ("sklearn_kmeans_dict_config", "sklearn_kmeans", SklearnKMeans),
        ("sklearn_dbscan_dict_config", "sklearn_dbscan", SklearnDBSCAN),
        (
            "sklearn_mean_shift_dict_config",
            "sklearn_mean_shift",
            SklearnMeanShift,
        ),
    ],
    indirect=True,
)
def test_container_sklearn_init(
    mock_dict_config: dict, attr_name: str, algo_class: AlgoSklearnType
) -> None:
    """Test the initialization of Container and initialization of sklearn
    instances with mocked configuration.

    Create instance of Container, setup configuration from mocked dict config,
    initialize attribute of Container with provided attr_name, check if
    attribute is instance of provided algo_class.

    Args:
        mock_dict_config (dict): Mock configuration for corresponding
        sklearn class.
        attr_name (str): Container attribute name for corresponding sklearn
        class.
        algo_class (AlgoSklearnType): Expected class of container attribute.

    Asserts:
        The container is initialized correctly without raising an error.
        The instance of sklearn class is initialized correctly.
        The instance of sklearn class is instance of corresponding class.
    """
    container = Container()
    container.config.from_dict(mock_dict_config)
    container_sklearn_attr_name = getattr(container, attr_name)

    sklearn_attr_name = container_sklearn_attr_name()
    assert isinstance(sklearn_attr_name, algo_class)


@pytest.mark.parametrize(
    "sklearn_class, attr_name, sklearn_attr_name, algo_class",
    [
        (SklearnKMeans, "kmeans", "sklearn_kmeans", KMeans),
        (SklearnDBSCAN, "dbscan", "sklearn_dbscan", DBSCAN),
        (SklearnMeanShift, "mean_shift", "sklearn_mean_shift", MeanShift),
    ],
)
def test_algo_init(
    sklearn_class: AlgoSklearnType,
    attr_name: str,
    sklearn_attr_name: str,
    algo_class: AlgoType,
) -> None:
    """Test that algorithm is initialized correctly with the
    dependency-injector container.

    Args:
        sklearn_class (AlgoSklearnType): Sklearn class.
        attr_name (tuple[str, str]): The attribute names.
        algo_class (AlgoType): The class object.

    Asserts:
        The algo_instance is instantiated without raising an error.
        The algo_instance is an instance of the expected class.
        The algo attribute of algo_instance is of the expected class.

    """
    mock_sklearn = Mock(spec=sklearn_class)
    container = Container()
    container_algo = getattr(container, attr_name)
    container_sklearn_algo = getattr(container, sklearn_attr_name)

    with container_sklearn_algo.override(mock_sklearn):
        algo_instance = container_algo()
        assert isinstance(algo_instance, algo_class)


@pytest.mark.parametrize(
    "algo_type, expected_type",
    [("kmeans", KMeans), ("dbscan", DBSCAN), ("mean_shift", MeanShift)],
)
def test_algorithm_selector(algo_type: str, expected_type: AlgoType) -> None:
    """Test that selected algorithm is selected correctly based on provided
    algorithm type from configuration.

    Args:
        algo_type (str): Type of algorithm to override configuration.
        expected_type (AlgoType): Expected type of selected algorithm instance.

    Asserts:
        Selected algorithm is initialized correctly and is an instance of the
        correct class.
    """
    container = Container()
    container.config.algorithm_type.override(algo_type)
    algorithm = container.algorithm()
    assert isinstance(algorithm, expected_type)
