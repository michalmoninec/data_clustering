import pytest

from typing import Type, Union
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.cluster import MeanShift as SklearnMeanShift

from src.clustering.utils.container import Container

AlgoSklearnType = Union[
    Type[SklearnKMeans], Type[SklearnDBSCAN], Type[SklearnMeanShift]
]


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
