import pytest
import numpy as np

from typing import Union, Type
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.cluster import MeanShift as SklearnMeanShift

from src.clustering.utils.algorithms import KMeans, DBSCAN, MeanShift

AlgoType = Union[Type[KMeans], Type[DBSCAN], Type[MeanShift]]
AlgoSklearnType = Union[
    Type[SklearnKMeans], Type[SklearnDBSCAN], Type[SklearnMeanShift]
]


@pytest.mark.parametrize(
    "algo_class, algo_sklearn_instance",
    [
        (KMeans, "sklearn_kmeans_instance"),
        (DBSCAN, "sklearn_dbscan_instance"),
        (MeanShift, "sklearn_mean_shift_instance"),
    ],
    indirect=True,
)
def test_algo_integration(
    algo_class: AlgoType,
    algo_sklearn_instance: AlgoSklearnType,
    mock_data: np.ndarray,
) -> None:
    """Test if the algorithm classes are initialized correctly with the
    the actual instance of sklearn.cluster algorithms.

    Args:
        algo_class (AlgoType): Algo class based on provided parameter.
        algo_sklearn_instance (AlgoSklearnType): Instance of the sklearn
            algo based on parameter.
        mock_data (np.ndarray): Mock test data.

    Asserts that:
        - The algo_class is initialized without raising an error.
        - The clustered_data has the same length as mock_data.
    """

    algo_instance = algo_class(algo_sklearn_instance)
    clustered_data = algo_instance.cluster_data(mock_data)

    assert mock_data.shape[0] == clustered_data.shape[0]
