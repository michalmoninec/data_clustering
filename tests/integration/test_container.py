import pytest

from typing import Type, Union
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.cluster import MeanShift as SklearnMeanShift
from pathlib import Path

from src.clustering.utils.container import Container
from src.clustering.utils.algorithms import KMeans, DBSCAN, MeanShift
from src.clustering.utils.input_handler import InputHandler
from src.clustering.utils.data_model import ConfigValidator

AlgoType = Union[Type[KMeans], Type[DBSCAN], Type[MeanShift]]
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


@pytest.mark.parametrize(
    "container, attr_name, algo_class, attr_class",
    [
        ("kmeans_config", "kmeans", KMeans, SklearnKMeans),
        ("dbscan_config", "dbscan", DBSCAN, SklearnDBSCAN),
        ("mean_shift_config", "mean_shift", MeanShift, SklearnMeanShift),
    ],
    indirect=True,
)
def test_algo_init(
    container: Container,
    attr_name: str,
    algo_class: AlgoType,
    attr_class: AlgoSklearnType,
) -> None:
    """Test that algorithm is initialized correctly with the
    dependency-injector container.

    Args:
        container (Container): The dependency injection container.
        attr_name (str): The attribute name.
        algo_class (AlgoType): The class type.
        attr_class (AlgoSklearnType): The attribute class type.

    Asserts:
        The algo_instance is initialized without raising an error.
        The algo_instance is an instance of the expected class.
        The algo attribute of algo_instance is of the expected class.

    """
    container_attr_name = getattr(container, attr_name)
    algo_instance = container_attr_name()

    assert isinstance(algo_instance, algo_class)
    assert isinstance(algo_instance.algo, attr_class)


@pytest.mark.parametrize(
    "container, algo_class",
    [
        ("kmeans_config", KMeans),
        ("dbscan_config", DBSCAN),
        ("mean_shift_config", MeanShift),
    ],
    indirect=True,
)
def test_selected_algo(
    container: Container,
    algo_class: AlgoType,
) -> None:
    """Test that algorithm is selected correctly with the
    dependency-injector container.

    Args:
        container (Container): The dependency injection container.
        algo_class (AlgoType): The class type.

    Asserts:
        The algorithm is initialized without raising an error.
        The algorithm is an instance of the expected class.

    """
    algorithm = container.algorithm()

    assert isinstance(algorithm, algo_class)


@pytest.mark.parametrize(
    "file_path, mock_yaml_config",
    [("numpy_path", "kmeans_config")],
    indirect=True,
)
def test_input_handler(file_path: str, mock_yaml_config: str) -> None:
    """Test input_handler instance initialization.

    Args:
        file_path (str): Path to input data file.
        mock_yaml_config (str): Path to mocked yaml config.

    Assert:
        The input_handler is initialized without raising an error.
        The input_handler is an instance of InputHandler.
        The path attribute of input_handler is an instance of Path.

    """
    container = Container()
    container.config.from_yaml(mock_yaml_config)
    container.config.input_data_path.override(file_path)

    input_handler = container.input_handler()
    assert isinstance(input_handler, InputHandler)
    assert isinstance(input_handler.path, Path)


@pytest.mark.parametrize(
    "mock_yaml_config",
    ["kmeans_config", "dbscan_config", "mean_shift_config"],
    indirect=True,
)
def test_general_validator(mock_yaml_config: str) -> None:
    """Test general_validator instance initialization.

    Args:
        mock_yaml_config (str): Path to yaml config file.

    Asserts:
        The general_validator is initialized without raising an error.
        The general_validator is an instance of ConfigValidator.
    """

    container = Container()
    container.config.from_yaml(mock_yaml_config)
    general_validator = container.general_validator()

    assert isinstance(general_validator, ConfigValidator)


@pytest.mark.parametrize(
    "mock_yaml_config",
    ["kmeans_config", "dbscan_config", "mean_shift_config"],
    indirect=True,
)
def test_algo_specific_validator(mock_yaml_config: str) -> None:
    """Test general_validator instance initialization.

    Args:
        mock_yaml_config (str): Path to yaml config file.

    Asserts:
        The general_validator is initialized without raising an error.
        The general_validator is an instance of ConfigValidator.
    """

    container = Container()
    container.config.from_yaml(mock_yaml_config)
    algo_specific_validator = container.algo_specific_validator()

    assert isinstance(algo_specific_validator, ConfigValidator)
