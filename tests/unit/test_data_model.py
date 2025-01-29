import pytest

from typing import Union, Type
from unittest.mock import Mock, patch
from pathlib import Path

from src.clustering.utils.data_model import (
    KMeansModel,
    DBSCANModel,
    MeanShiftModel,
    ConfigValidator,
)

ModelType = Union[Type[KMeansModel], Type[DBSCANModel], Type[MeanShiftModel]]


@pytest.mark.parametrize(
    "model_class", [KMeansModel, DBSCANModel, MeanShiftModel]
)
def test_validator_init(model_class: ModelType) -> None:
    mock_model = Mock(spec=model_class)
    config_validator = ConfigValidator(model=mock_model)
    assert isinstance(config_validator.model, model_class)


@pytest.mark.parametrize(
    "mock_dict_config, model_class",
    [
        ("sklearn_kmeans_dict_config", KMeansModel),
        ("sklearn_dbscan_dict_config", DBSCANModel),
        ("sklearn_mean_shift_dict_config", MeanShiftModel),
    ],
    indirect=True,
)
def test_invalid_data_model(
    mock_dict_config: dict, model_class: ModelType
) -> None:
    """Test that data validation based on provided model is handled correctly
    for invalid data.

    Args:
        mock_dict_config (dict): Mock configuration for corresponding
        sklearn class.
        model_class (ModelType): Class of corresponding ModelType.

    Asserts:
        The TypeError is raised.
        Mocked_open is called once with correct argument.
        Mocked_load is called once with correct argument.
    """
    mock_dict_config["algorithm_type"] = "non_existent"
    mock_path = Mock(spec=Path)
    config_validator = ConfigValidator(model=model_class)
    with (
        patch("builtins.open") as mocked_open,
        patch("yaml.safe_load", return_value=mock_dict_config) as mocked_load,
    ):
        with pytest.raises(TypeError):
            config_validator.validate_data(mock_path)
            mocked_open.assert_called_once_with(mock_path, "r")
            mocked_load.assert_called_once_with(mocked_open().__enter__())


@pytest.mark.parametrize(
    "mock_dict_config, model_class",
    [
        ("sklearn_kmeans_dict_config", KMeansModel),
        ("sklearn_dbscan_dict_config", DBSCANModel),
        ("sklearn_mean_shift_dict_config", MeanShiftModel),
    ],
    indirect=True,
)
def test_valid_data_model(
    mock_dict_config: dict, model_class: ModelType
) -> None:
    """Test that data validation based on provided model is handled correctly
    for valid data.

    Args:
        mock_dict_config (dict): Mock configuration for corresponding
        sklearn class.
        model_class (ModelType): Class of corresponding ModelType.

    Asserts:
        The data is validated without raising an error.
    """
    mock_path = Mock(spec=Path)
    config_validator = ConfigValidator(model=model_class)
    with (
        patch("builtins.open") as mocked_open,
        patch("yaml.safe_load", return_value=mock_dict_config) as mocked_load,
    ):
        config_validator.validate_data(mock_path)
        mocked_open.assert_called_once_with(mock_path, "r")
        mocked_load.assert_called_once_with(mocked_open().__enter__())
