import pytest
import numpy as np

from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from unittest.mock import Mock
from typing import Type, Union

from src.clustering.utils.algorithms import KMeans, DBSCAN

AlgoType = Union[Type[KMeans], Type[DBSCAN]]
AlgoSklearnType = Union[Type[SklearnKMeans], Type[SklearnDBSCAN]]


@pytest.mark.parametrize(
    "sklearn_class, algo_class",
    [
        (SklearnKMeans, KMeans),
        (SklearnDBSCAN, DBSCAN),
    ],
)
def test_algo_init(
    sklearn_class: AlgoSklearnType, algo_class: AlgoType
) -> None:
    """Test the initialization of KMeans class.

    Validates that the __init__ method correctly initializes the algo
    attribute with the provided algorithm instance.

    Args:
        sklearn_class (AlgoSklearnType): Sklearn algo class based on provided
        parameter.
        algo_class (AlgoType): Algo class based on provided parameter.

    Asserts that:
        - The algo_class is initialized without raising an error.
        - The algo attribute is set correctly.
        - The algo attribute is an instance of sklearn_class.

    """
    mock_sklearn = Mock(spec=sklearn_class)
    algo_instance = algo_class(mock_sklearn)

    assert algo_instance.algo == mock_sklearn
    assert isinstance(algo_instance.algo, sklearn_class)


@pytest.mark.parametrize(
    "sklearn_class, algo_class",
    [
        (SklearnKMeans, KMeans),
        (SklearnDBSCAN, DBSCAN),
    ],
)
def test_algo_cluster_empty_data(
    sklearn_class: AlgoSklearnType, algo_class: AlgoType
) -> None:
    """Test the `cluster_data` method of algo_class.

    Validates that the `cluster_data` method correctly handles empty array data
    input.

    Args:
        sklearn_class (AlgoSklearnType): Mocked algo based on provided
        parameter.
        algo_class (AlgoType): Algo class based on provided parameter.

    Asserts that:
        -   The size of clustered_data is zero.

    """
    mock_sklearn = Mock(spec=sklearn_class)
    if sklearn_class == SklearnDBSCAN:
        mock_sklearn.fit_predict.return_value = np.array([1, 1, 1, 0, 0, 0])
    else:
        mock_sklearn.predict.return_value = np.array([1, 1, 1, 0, 0, 0])

    algo_instance = algo_class(mock_sklearn)
    clustered_data = algo_instance.cluster_data(np.array([]))
    assert clustered_data.size == 0


@pytest.mark.parametrize(
    "sklearn_class, algo_class",
    [
        (SklearnKMeans, KMeans),
        (SklearnDBSCAN, DBSCAN),
    ],
)
def test_algo_cluster_valid_input_data(
    sklearn_class: AlgoSklearnType, algo_class: AlgoType, mock_data: np.ndarray
) -> None:
    """Test the `cluster_data` method of algo_class.

    Validates that the `cluster_data` method correctly handles valid array data
    input.

    Args:
        sklearn_class (AlgoSklearnType): Sklearn algo class based on provided
        parameter.
        algo_class (AlgoType): Algo class based on provided parameter.
        mock_data (np.ndarray): Mock test data.

    Asserts that:
        -   The `fit` and `predict` method of mock_sklearn is called once
        with corresponding data.
    """
    mock_predict_value = np.array([1, 1, 1, 0, 0, 0])
    mock_sklearn = Mock(spec=sklearn_class)
    if sklearn_class == SklearnDBSCAN:
        mock_sklearn.fit_predict.return_value = mock_predict_value
    else:
        mock_sklearn.predict.return_value = mock_predict_value

    algo_instance = algo_class(mock_sklearn)
    algo_instance.cluster_data(mock_data)

    mock_sklearn.fit.assert_called_once_with(mock_data)

    if sklearn_class == SklearnDBSCAN:
        mock_sklearn.fit_predict.asssert_called_once_with(mock_data)
    else:
        mock_sklearn.predict.asssert_called_once_with(mock_data)


@pytest.mark.parametrize(
    "sklearn_class, algo_class",
    [
        (SklearnKMeans, KMeans),
        (SklearnDBSCAN, DBSCAN),
    ],
)
def test_algo_cluster_append_labels(
    sklearn_class: AlgoSklearnType, algo_class: AlgoType, mock_data: np.ndarray
) -> None:
    """Test the `cluster_data` method of algo_class.

    Validates that the `cluster_data` method correctly append labels to input
    data array.

    Args:
        sklearn_class (AlgoSklearnType): Sklearn algo class based on provided
        parameter.
        algo_class (AlgoType): Algo class based on provided parameter.
        mock_data (np.ndarray): Mock test data.

    Asserts that:
        -   The output is the data with cluster labels appended.
    """
    mock_sklearn = Mock(spec=sklearn_class)
    mock_predict_value = np.array([1, 1, 1, 0, 0, 0])
    if sklearn_class == SklearnDBSCAN:
        mock_sklearn.fit_predict.return_value = mock_predict_value
    else:
        mock_sklearn.predict.return_value = mock_predict_value

    expected_result = np.hstack((mock_data, mock_predict_value.reshape(-1, 1)))

    algo_instance = algo_class(mock_sklearn)
    clustered_data = algo_instance.cluster_data(mock_data)

    assert np.array_equal(clustered_data, expected_result)
