from pytest import fixture, FixtureRequest


@fixture
def sklearn_kmeans_dict_config() -> dict:
    """Fixture, that creates and returns dict config for KMeans."""

    return {
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
