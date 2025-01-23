import pytest

from typing import Type, Union
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.cluster import MeanShift as SklearnMeanShift

from clustering.utils.container import Container

AlgoSklearnType = Union[
    Type[SklearnKMeans], Type[SklearnDBSCAN], Type[SklearnMeanShift]
]


@pytest.mark.parametrize(
    "mock_yaml_config",
    [
        ("kmeans_config"),
        ("dbscan_config"),
        ("mean_shift_config"),
    ],
    indirect=True,
)
def test_container_config_init(mock_yaml_config: str) -> None:
    """Test the initialization of Container and initialization of config
    instance with mocked configuration yaml file.

    Args:
        mock_yaml_config (str): Mock configuration for corresponding
        sklearn class.
        attr_name (str): Container attribute name for corresponding sklearn
        class.

    Asserts:
        The container is initialized correctly without raising an error.
        The instance of container config is initialized correctly.
    """
    container = Container()
    container.config.from_yaml(mock_yaml_config)
    container.config()


@pytest.mark.parametrize(
    "container, attr_name, algo_class",
    [
        ("kmeans_config", "sklearn_kmeans", SklearnKMeans),
        ("dbscan_config", "sklearn_dbscan", SklearnDBSCAN),
        ("mean_shift_config", "sklearn_mean_shift", SklearnMeanShift),
    ],
    indirect=True,
)
def test_sklearn_init(
    container: Container,
    attr_name: str,
    algo_class: AlgoSklearnType,
) -> None:
    """Test that the sklearn algorithm is initialized correctly with the
    dependency-injector container.

    Args:
        container (Container): The dependency injection container.
        attr_name (str): The attribute name.
        algo_class (AlgoSklearnType): Expected class of container attribute.

    Asserts:
        The algorithm is initialized without raising an error.
        The algorithm is an instance of the expected class.
    """
    container_sklearn_attr_name = getattr(container, attr_name)
    sklearn_attr_name = container_sklearn_attr_name()

    assert isinstance(sklearn_attr_name, algo_class)
