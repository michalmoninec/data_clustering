from pytest import fixture, FixtureRequest


@fixture
def sklearn_kmeans_dict_config() -> dict:
    """Fixture, that creates and returns dict config for KMeans."""

    return {
        "algorithm_type": "kmeans",
        "input_data_path": "input_data.npy",
        "output_data_format": "numpy",
        "kmeans": {
            "n_clusters": 2,
            "random_state": False,
            "max_iter": 250,
            "init": "k-means++",
        },
    }


@fixture
def sklearn_dbscan_dict_config() -> dict:
    """Fixture, that creates and returns dict config for DBSCAN."""

    return {
        "algorithm_type": "dbscan",
        "input_data_path": "input_data.npy",
        "output_data_format": "numpy",
        "dbscan": {
            "eps": 0.5,
            "min_samples": 5,
            "algorithm": "auto",
            "leaf_size": 30,
        },
    }


@fixture
def sklearn_mean_shift_dict_config() -> dict:
    """Fixture, that creates and returns dict config for MeanShift."""

    return {
        "algorithm_type": "mean_shift",
        "input_data_path": "input_data.npy",
        "output_data_format": "numpy",
        "mean_shift": {
            "bandwidth": 0.5,
            "seeds": None,
            "bin_seeding": False,
            "min_bin_freq": 1,
            "cluster_all": True,
            "max_iter": 300,
        },
    }


@fixture
def mock_dict_config(request: FixtureRequest) -> dict:
    """Fixture, that returns value of fixture for provided name."""
    return request.getfixturevalue(request.param)
