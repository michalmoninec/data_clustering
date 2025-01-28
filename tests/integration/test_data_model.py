import pytest

from typing import Union, Type

from src.clustering.utils.data_model import (
    KMeansModel,
    DBSCANModel,
    MeanShiftModel,
    ConfigValidator,
)

ModelType = Union[Type[KMeansModel], Type[DBSCANModel], Type[MeanShiftModel]]


@pytest.mark.parametrize(
    "mock_yaml_config, model_class",
    [
        ("invalid_kmeans_config", KMeansModel),
        ("invalid_dbscan_config", DBSCANModel),
        ("invalid_mean_shift_config", MeanShiftModel),
    ],
    indirect=True,
)
def test_invalid_data(mock_yaml_config: str, model_class: ModelType) -> None:
    """Test that data validation based on provided model is handled correctly
    for valid data.

    Args:
        mock_yaml_config (dict): Mock configuration for corresponding
        sklearn class.
        model_class (ModelType): Class of corresponding ModelType.

    Asserts:
        The TypeError is raised.
    """
    config_validator = ConfigValidator(model=model_class)

    with pytest.raises(TypeError):
        config_validator.validate_data(mock_yaml_config)


@pytest.mark.parametrize(
    "mock_yaml_config, model_class",
    [
        ("kmeans_config", KMeansModel),
        ("dbscan_config", DBSCANModel),
        ("mean_shift_config", MeanShiftModel),
    ],
    indirect=True,
)
def test_valid_data(mock_yaml_config: str, model_class: ModelType) -> None:
    """Test that data validation based on provided model is handled correctly
    for valid data.

    Args:
        mock_yaml_config (dict): Mock configuration for corresponding
        sklearn class.
        model_class (ModelType): Class of corresponding ModelType.

    Asserts:
        The data is validated without raising an error.
    """
    config_validator = ConfigValidator(model=model_class)
    config_validator.validate_data(mock_yaml_config)
