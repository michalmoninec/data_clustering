from dependency_injector import containers, providers
from pathlib import Path

from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.cluster import MeanShift as SklearnMeanShift

from src.clustering.utils.algorithms import KMeans, DBSCAN, MeanShift
from src.clustering.utils.input_handler import InputHandler
from src.clustering.utils.data_model import (
    ConfigModel,
    ConfigValidator,
    KMeansModel,
    DBSCANModel,
    MeanShiftModel,
)
from src.clustering.utils.output_handler import (
    NumpyOutputHandler,
    JSONOutputHandler,
)


class Container(containers.DeclarativeContainer):
    """
    Dependency injection container for managing and providing
    application dependencies.

    Attributes:
        `config` (providers.Configuration):
            Initializes the configuration settings by loading values from
            the specified YAML file(s). In this example, it is configured
            to read from `config.yaml`.

        `sklearn_kmeans` (providers.Singleton):
            Initializes the KMeans algorithm using the scikit-learn
            implementation. The algorithm is configured with the following
            settings:
            - `n_clusters`: The number of clusters to form.
            - `random_state`: The seed used by the random number generator.
            - `max_iter`: The maximum number of iterations for the algorithm.
            - `init`: The method used to initialize the centroids.

        `sklearn_dbscan` (providers.Singleton):
            Initializes the DBSCAN algorithm using the scikit-learn
            implementation. The algorithm is configured with the following
            settings:
            - `eps`: The maximum distance between two samples for them to be
            considered as in the same neighborhood.
            - `min_samples`: The number of samples in a neighborhood for a
            point to be considered as a core point.
            - `algorithm`: The algorithm used to compute the nearest neighbors.
            - `leaf_size`: The leaf size of the KD-tree.

        `sklearn_mean_shift` (providers.Singleton):
            Initializes the MeanShift algorithm using the scikit-learn
            implementation. The algorithm is configured with the following
            settings:
            - `bandwidth`: The bandwidth used in the kernel density estimation.
            - `bin_seeding`: Whether to use bin seeding to initialize the
            centroids.
            - `min_bin_freq`: The minimum number of samples in a bin to
            consider it as a seed.
            - `cluster_all`: Whether to cluster all points or only the seeds.
            - `max_iter`: The maximum number of iterations for the algorithm.

        `kmeans` (providers.Singleton):
            Initializes KMeans algorithm with provided instance of
            SKlearnKMeans

        `dbscan` (providers.Singleton):
            Initializes DBSCAN algorithm with provided instance of
            SklearnDBSCAN.

        `mean_shift` (providers.Singleton):
            Initializes MeanShift algorithm with provided instance of
            SklearnMeanShift.

        `algorithm` (providers.Selector):
            Select concrete clustering algorithm based on configuration.

        `input_handler` (providers.Singleton):
            Initializes InputHandler with path based on configuration.

        `general_validator` (providers.Singleton):
            Initializes InputHandler with ConfigModel, that validates the
            general part of yaml file.

        `algo_specific_validator` (providers.Singleton):
            Initializes InputHandler with specific algorithm model, that
            validates the whole yaml file.

        `output_handler` (providers.Selector):
            Select concrete OutputHandler based on configuration.
    """

    config = providers.Configuration(yaml_files=["config.yaml"])

    sklearn_kmeans = providers.Singleton(
        SklearnKMeans,
        n_clusters=config.kmeans.n_clusters,
        random_state=config.kmeans.random_state,
        max_iter=config.kmeans.max_iter,
        init=config.kmeans.init,
    )

    sklearn_dbscan = providers.Singleton(
        SklearnDBSCAN,
        eps=config.dbscan.eps,
        min_samples=config.dbscan.min_samples,
        algorithm=config.dbscan.algorithm,
        leaf_size=config.dbscan.leaf_size,
    )

    sklearn_mean_shift = providers.Singleton(
        SklearnMeanShift,
        bandwidth=config.mean_shift.bandwidth,
        bin_seeding=config.mean_shift.bin_seeding,
        min_bin_freq=config.mean_shift.min_bin_freq,
        cluster_all=config.mean_shift.cluster_all,
        max_iter=config.mean_shift.max_iter,
    )

    kmeans = providers.Singleton(
        KMeans,
        algo=sklearn_kmeans,
    )

    dbscan = providers.Singleton(
        DBSCAN,
        algo=sklearn_dbscan,
    )

    mean_shift = providers.Singleton(
        MeanShift,
        algo=sklearn_mean_shift,
    )

    algorithm = providers.Selector(
        config.algorithm_type,
        kmeans=kmeans,
        dbscan=dbscan,
        mean_shift=mean_shift,
    )

    input_handler = providers.Singleton(
        InputHandler,
        path=providers.Singleton(Path, config.input_data_path),
    )

    general_validator = providers.Singleton(
        ConfigValidator, model=providers.Object(ConfigModel)
    )

    algo_specific_validator = providers.Singleton(
        ConfigValidator,
        model=providers.Selector(
            config.algorithm_type,
            kmeans=providers.Object(KMeansModel),
            dbscan=providers.Object(DBSCANModel),
            mean_shift=providers.Object(MeanShiftModel),
        ),
    )

    output_handler = providers.Selector(
        config.output_data_format,
        numpy=providers.Singleton(NumpyOutputHandler),
        json=providers.Singleton(JSONOutputHandler),
    )
