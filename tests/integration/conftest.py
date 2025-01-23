import os
import tempfile

from pytest import fixture, FixtureRequest
from typing import Generator, Union, Type
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.cluster import MeanShift as SklearnMeanShift

from src.clustering.utils.container import Container
from src.clustering.utils.algorithms import BaseAlgo, KMeans, DBSCAN, MeanShift

AlgoSklearnType = Union[
    Type[SklearnKMeans], Type[SklearnDBSCAN], Type[SklearnMeanShift]
]


@fixture
def algo_sklearn_instance(request: FixtureRequest) -> AlgoSklearnType:
    """Fixture that returns specific fixture, based on provided parameter.

    Args:
        request (FixtureRequest): Pytest fixture request object.

    Returns:
        AlgoSklearnType: Specific fixture with correspondend sklearn instance.
    """

    return request.getfixturevalue(request.param)


@fixture
def sklearn_kmeans_instance() -> SklearnKMeans:
    """Fixture that creates and sets up a mock object of
    the SklearnKMeans class.

    Returns:
        SklearnKMeans: A mock object of the SklearnKMeans class.
    """
    return SklearnKMeans(
        n_clusters=3, random_state=0, max_iter=300, init="k-means++"
    )


@fixture
def sklearn_dbscan_instance() -> SklearnDBSCAN:
    """Fixture that creates and sets up a mock object of
    the SklearnDBSCAN class.

    Returns:
        SklearnDBSCAN: A mock object of the SklearnKMeans class.
    """
    return SklearnDBSCAN(
        eps=0.5,
        min_samples=5,
        algorithm="auto",
        leaf_size=30,
    )


@fixture
def sklearn_mean_shift_instance() -> SklearnMeanShift:
    """Fixture that creates and sets up a mock object of
    the SklearnMeanShift class.

    Returns:
        SklearnMeanShift: A mock object of the SklearnKMeans class.
    """
    return SklearnMeanShift(
        bandwidth=0.5,
        seeds=None,
        bin_seeding=False,
        min_bin_freq=1,
        cluster_all=True,
        max_iter=300,
    )


@fixture
def algo_instance(request: FixtureRequest) -> BaseAlgo:
    """Fixture that returns specific fixture, based on provided parameter.

    Args:
        request (FixtureRequest): Pytest fixture request object.

    Returns:
        BaseAlgo: Specific fixture.
    """

    return request.getfixturevalue(request.param)


@fixture
def kmeans_instance(sklearn_kmeans_instance) -> KMeans:
    """Fixture that creates and sets up an object of the KMeans class.

    Returns:
        KMeans: An instance of the KMeans class.
    """
    return KMeans(sklearn_kmeans_instance)


@fixture
def dbscan_instance(sklearn_dbscan_instance) -> DBSCAN:
    """Fixture that creates and sets up an object of the DBSCAN class.

    Returns:
        DBSCAN: An instance of the DBSCAN class.
    """
    return DBSCAN(sklearn_dbscan_instance)


@fixture
def mean_shift_instance(sklearn_mean_shift_instance) -> MeanShift:
    """Fixture that creates and sets up an object of the MeanShift class.

    Returns:
        MeanShift: An instance of the MeanShift class.
    """
    return MeanShift(sklearn_mean_shift_instance)


@fixture
def kmeans_config() -> Generator[str, None, None]:
    """Fixture that creates temporary yaml config file and
    yields a file path with KMeans configuration settings.

    yields:
        str: The KMeans configuration settings.
    """

    config_data = """
    algorithm_type: kmeans

    kmeans:
        n_clusters: 2
        random_state: false
        max_iter: 250
        init: "k-means++"
    """
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".yaml", delete=False
    ) as temp_file:
        temp_file.write(config_data)
        temp_file.close()
        try:
            yield temp_file.name
        finally:
            os.remove(temp_file.name)


@fixture
def dbscan_config() -> Generator[str, None, None]:
    """Fixture that creates temporary yaml config file and
    yields a file path with DBSCAN configuration settings.

    Returns:
        str: The DBSCAN configuration settings.
    """

    config_data = """
    algorithm_type: dbscan

    dbscan:
        eps: 0.5
        min_samples: 5
        algorithm: auto
        leaf_size: 30
    """
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".yaml", delete=False
    ) as temp_file:
        temp_file.write(config_data)
        temp_file.close()
        try:
            yield temp_file.name
        finally:
            os.remove(temp_file.name)


@fixture
def mean_shift_config() -> Generator[str, None, None]:
    """Fixture that creates temporary yaml config file and
    yields a file path with MeanShift configuration settings.

    Yields:
        str: The MeanShift configuration settings.
    """

    config_data = """
    algorithm_type: mean_shift

    mean_shift:
        bandwidth: 0.5
        seeds: none
        bin_seeding: false
        min_bin_freq: 1
        cluster_all: true
        max_iter: 300
    """
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".yaml", delete=False
    ) as temp_file:
        temp_file.write(config_data)
        temp_file.close()
        try:
            yield temp_file.name
        finally:
            os.remove(temp_file.name)


@fixture
def mock_yaml_config(request: FixtureRequest) -> Generator[str, None, None]:
    """Fixture, that returns value of fixture for provided name."""
    return request.getfixturevalue(request.param)


@fixture
def container(request: FixtureRequest) -> Container:
    """Fixture that returns a Container object, with configuration based on
    provided parameter.

    Args:
        request (FixtureRequest): The request object.

    Returns:
        Container: The Container object with the specified configuration.
    """
    config_file = request.getfixturevalue(request.param)
    container = Container()
    container.config.from_yaml(config_file)
    return container
