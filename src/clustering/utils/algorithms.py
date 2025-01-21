import numpy as np

from abc import ABC, abstractmethod
from sklearn.cluster import KMeans as SklearnKMeans


class BaseAlgo(ABC):
    """Abstract base class for clustering algorithms."""

    @abstractmethod
    def cluster_data(self, data: np.ndarray) -> np.ndarray:
        """Cluster the provided data.

        This method must be implemented by subclasses.

        Args:
            data (np.ndarray): The data to cluster.

        Returns:
            data (np.ndarray): The data with cluster labels appended.
        """
        pass


class KMeans(BaseAlgo):
    """Implementation of the BaseAlgo class using KMeans from scikit-learn.

    This class provides an implementation of the `BaseAlgo` interface
    using the KMeans algorithm from scikit-learn. It wraps the KMeans
    functionality to provide `cluster_data` method using `fit` and `predict`
    methods of the algo instance.

    Attributes:
        algo (SklearnKMeans): An instance of scikit-learn's KMeans.
    """

    def __init__(self, algo: SklearnKMeans) -> None:
        """Initializes the KMeans.

        Args:
            algo (SklearnKMeans): An instance of scikit-learn's KMeans.
        """
        self.algo = algo

    def cluster_data(self, data: np.ndarray) -> np.ndarray:
        """Cluster the provided data using the algorithm.

        This method fits the algorithm to the provided data, creates labels
        for the data and appends the cluster labels to the data. Empty data
        is returned as is.

        Args:
            data (np.ndarray): The data to cluster.

        Returns:
            np.ndarray: The data with cluster labels appended.
        """
        if data.size == 0:
            return data

        self.algo.fit(data)
        labels = self.algo.predict(data)
        return np.hstack((data, labels.reshape(-1, 1)))
