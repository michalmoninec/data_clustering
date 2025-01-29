import yaml

from pydantic import BaseModel, Field, ValidationError
from typing import Optional, Literal, List, Union


class ConfigModel(BaseModel):
    """
    Configuration model for a clustering algorithm.

    Attributes:
        algorithm_type (Literal["kmeans", "dbscan", "mean_shift"]):
            The type of clustering algorithm to use.
        input_data_path (str):
            The file path to the input data.
        output_data_format (Literal["numpy", "csv", "json"]):
            The format in which the output data should be saved.
    """

    algorithm_type: Literal["kmeans", "dbscan", "mean_shift"]
    input_data_path: str
    output_data_format: Literal["numpy", "csv", "json"]


class KMeansParamsConfig(BaseModel):
    """
    Configuration model for K-Means clustering parameters.

    Attributes:
        n_clusters (int):
            The number of clusters to form. Must be greater than 0.
        random_state (Optional[int]):
            Seed for the random number generator to ensure reproducibility.
            If None, randomness is not controlled.
        max_iter (int):
            Maximum number of iterations for the K-Means algorithm.
            Must be greater than 0.
        init (Literal["random", "k-means++"]):
            Initialization method for cluster centroids.
            - "random": Selects initial cluster centers randomly.
            - "k-means++": Uses a smart seeding technique to improve
            convergence.
    """

    n_clusters: int = Field(..., gt=0)
    random_state: Optional[int]
    max_iter: int = Field(..., gt=0)
    init: Literal["random", "k-means++"]


class DBSCANParamsConfig(BaseModel):
    """
    Configuration model for DBSCAN clustering parameters.

    Attributes:
        eps (float):
            The maximum distance between two samples for them to be
            considered as neighbors.
            Must be greater than 0.
        min_samples (int):
            The minimum number of points required to form a dense
            region (core point).
            Must be at least 1.
        algorithm (Literal["auto", "ball_tree", "kd_tree", "brute"]):
            The algorithm to use for computing nearest neighbors:
            - "auto": Chooses the best algorithm automatically.
            - "ball_tree": Uses a BallTree data structure.
            - "kd_tree": Uses a KDTree data structure.
            - "brute": Uses a brute-force approach.
        leaf_size (int):
            Leaf size for BallTree or KDTree algorithms.
            Must be greater than 1. Affects speed and memory consumption.
    """

    eps: float = Field(..., gt=0)
    min_samples: int = Field(..., ge=1)
    algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"]
    leaf_size: int = Field(..., gt=1)


class MeanShiftParamsConfig(BaseModel):
    """
    Configuration model for Mean Shift clustering parameters.

    Attributes:
        bandwidth (float):
            The radius of the area used to define clusters.
            Must be greater than 0.
        seeds (Union[Optional[List[List[float]]], Literal["none"]]):
            Initial seed points for clustering. Can be:
            - A list of coordinates representing seed points.
            - "none" to let the algorithm generate them automatically.
        bin_seeding (bool):
            If True, initial kernel locations are placed at the centers of
            data points
            rather than at random positions.
        min_bin_freq (int):
            The minimum number of points required in a bin to be considered a
            cluster center.
            Must be at least 1.
        cluster_all (bool):
            If True, assigns all points to a cluster; otherwise, leaves some
            as outliers.
        max_iter (int):
            The maximum number of iterations allowed for convergence.
            Must be at least 1.
    """

    bandwidth: float = Field(..., gt=0)
    seeds: Union[Optional[List[List[float]]], Literal["none"]]
    bin_seeding: bool
    min_bin_freq: int = Field(..., ge=1)
    cluster_all: bool
    max_iter: int = Field(..., ge=1)


class KMeansModel(ConfigModel):
    """
    Configuration model for the K-Means clustering algorithm.

    Attributes:
        algorithm_type (Literal["kmeans"]):
            Specifies that this configuration is for the K-Means algorithm.
        kmeans (KMeansParamsConfig):
            The parameters specific to the K-Means clustering algorithm.
    """

    algorithm_type: Literal["kmeans"]
    kmeans: KMeansParamsConfig


class DBSCANModel(ConfigModel):
    """
    Configuration model for the DBSCAN clustering algorithm.

    Attributes:
        algorithm_type (Literal["dbscan"]):
            Specifies that this configuration is for the DBSCAN algorithm.
        dbscan (DBSCANParamsConfig):
            The parameters specific to the DBSCAN clustering algorithm.
    """

    algorithm_type: Literal["dbscan"]
    dbscan: DBSCANParamsConfig


class MeanShiftModel(ConfigModel):
    """
    Configuration model for the Mean Shift clustering algorithm.

    Attributes:
        algorithm_type (Literal["mean_shift"]):
            Specifies that this configuration is for the Mean Shift algorithm.
        mean_shift (MeanShiftParamsConfig):
            The parameters specific to the Mean Shift clustering algorithm.
    """

    algorithm_type: Literal["mean_shift"]
    mean_shift: MeanShiftParamsConfig


class ConfigValidator:
    """Config file validator for correct schema of yaml file.

    Attributes:
        `model` (BaseModel): Pydantic model for corresponding algorithm
        configuration
    """

    def __init__(self, model: BaseModel) -> None:
        """Initializes the ConfigValidator.

        Args:
            model (BaseModel): Pydantic model for corresponding algorithm
            configuration.
        """
        self.model = model

    def validate_data(self, config_path: str) -> None:
        """Validates the provided file at config_path with corresponding
        algorithm configuration schema.

        Args:
            config_path (str): Path to config yaml file.

        Raises:
            TypeError: If data is not validated correctly.
        """
        with open(config_path, "r") as yaml_file:
            data = yaml.safe_load(yaml_file)
        try:
            self.model(**data)
        except ValidationError:
            raise TypeError("Config file is not correct.")
