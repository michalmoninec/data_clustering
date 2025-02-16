import pytest

from typing import Type, Union
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.cluster import MeanShift as SklearnMeanShift
from unittest.mock import Mock
from pathlib import Path

from src.clustering.utils.container import Container
from src.clustering.utils.algorithms import KMeans, DBSCAN, MeanShift
from src.clustering.utils.input_handler import InputHandler
from src.clustering.utils.data_model import (
    KMeansModel,
    DBSCANModel,
    MeanShiftModel,
    ConfigValidator,
)
from src.clustering.utils.output_handler import (
    NumpyOutputHandler,
    JSONOutputHandler,
)

AlgoSklearnType = Union[
    Type[SklearnKMeans], Type[SklearnDBSCAN], Type[SklearnMeanShift]
]
AlgoType = Union[Type[KMeans], Type[DBSCAN], Type[MeanShift]]
ModelType = Union[Type[KMeansModel], Type[DBSCANModel], Type[MeanShiftModel]]
HandlerType = Union[Type[NumpyOutputHandler], Type[JSONOutputHandler]]


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


def test_input_handler() -> None:
    """Test input_handled instance initialization.

    Assert:
        The input_handler is initialized without raising an error.
        The input_handler is an instance of InputHandler.
        The path attribute of input_handler is an instance of Path.

    """
    mock_path = "test.txt"

    container = Container()
    container.config.input_data_path.override(mock_path)

    input_handler = container.input_handler()
    assert isinstance(input_handler, InputHandler)
    assert isinstance(input_handler.path, Path)


@pytest.mark.parametrize(
    "mock_dict_config",
    [
        "sklearn_kmeans_dict_config",
        "sklearn_dbscan_dict_config",
        "sklearn_mean_shift_dict_config",
    ],
    indirect=True,
)
def test_general_validator(mock_dict_config: dict) -> None:
    """Test general_validator instance initialization.

    Args:
        mock_dict_config (dict): Mock dict configuration.

    Asserts:
        The general_validator is initialized without raising an error.
        The general_validator is an instance of ConfigValidator.
    """
    container = Container()
    container.config.from_dict(mock_dict_config)
    general_validator = container.general_validator()

    assert isinstance(general_validator, ConfigValidator)


@pytest.mark.parametrize(
    "mock_dict_config, model_class",
    [
        ("sklearn_kmeans_dict_config", KMeansModel),
        ("sklearn_dbscan_dict_config", DBSCANModel),
        ("sklearn_mean_shift_dict_config", MeanShiftModel),
    ],
    indirect=True,
)
def test_algo_specific_validator(
    mock_dict_config: dict, model_class: ModelType
) -> None:
    """Test algo_specific_validator instance initialization.

    Args:
        mock_dict_config (dict): Mock dict configuration.
        model_class (ModelType): Pydantic schema model.

    Asserts:
        The algo_specific_validator is initialized without raising an error.
        The algo_specific_validator is an instance of ConfigValidator.
    """
    container = Container()
    container.config.from_dict(mock_dict_config)

    algo_specific_validator = container.algo_specific_validator()

    assert isinstance(algo_specific_validator, ConfigValidator)


@pytest.mark.parametrize(
    "handler_class, output_format",
    [(NumpyOutputHandler, "numpy"), (JSONOutputHandler, "json")],
)
def test_output_handler(handler_class: HandlerType, output_format: str):
    """Test output_handler instance initialization based on provided config
    and selected output format.

    Args:
        handler_class (HandlerType): Concrete handler class.
        output_format (str): Output path format.

    Asserts:
        The output_handler is initialized without raising an error.
        The output_handler is an instance of handler_class.
    """
    container = Container()
    container.config.output_data_format.override(output_format)
    output_handler = container.output_handler()

    assert isinstance(output_handler, handler_class)
