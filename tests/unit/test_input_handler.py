import pytest

from unittest.mock import patch, Mock
from pathlib import Path

from src.clustering.utils.input_handler import InputHandler


def test_input_handler_init() -> None:
    """Test that the InputHandler is initialized correctly.

    Asserts:
        The InputHandler is initialized correctly without raising an error.
        The `path` attribute is of correct type.
    """
    mock_path = Mock(spec=Path)
    input_handler = InputHandler(path=mock_path)
    assert isinstance(input_handler.path, Path)


def test_load_data_non_existent_file() -> None:
    """Test that `load_data` method handles non existent file correctly.

    Args:
        mock_path (Mock): Mock path.

    Asserts:
        FileExistsError is raised.
    """
    mock_path = Mock(spec=Path)
    mock_path.exists.return_value = False
    input_handler = InputHandler(path=mock_path)
    with pytest.raises(FileExistsError):
        input_handler.load_data()


def test_load_data_invalid_suffix() -> None:
    """Test that `load_data` method handles invalid suffix correctly.

    Asserts:
        ValueError is raised.
    """
    mock_path = Mock(spec=Path)
    mock_path.exists.return_value = True
    mock_path.suffix.return_value = ".non_exist"
    input_handler = InputHandler(path=mock_path)
    with pytest.raises(ValueError):
        input_handler.load_data()


def test_load_data_numpy_suffix() -> None:
    """Test that `load_data` method handles numpy suffix correctly.

    Asserts:
        The mocked numpy load function was called once with correct data.
    """
    mock_path = Mock(spec=Path)
    mock_path.exists.return_value = True
    mock_path.suffix = ".npy"
    input_handler = InputHandler(path=mock_path)

    with patch("numpy.load") as mocked_numpy_load:
        input_handler.load_data()
        mocked_numpy_load.assert_called_once_with(mock_path)


def test_load_data_json_suffix() -> None:
    """Test that `load_data` method handles json file correctly.

    Asserts:
        The mocked open function is called once with correct data.
        The mocked json loads method is called once.
        The mocked nupmy load method was called once.
    """
    mock_path = Mock(spec=Path)
    mock_path.exists.return_value = True
    mock_path.suffix = ".json"
    input_handler = InputHandler(path=mock_path)

    with (
        patch("builtins.open") as mocked_open,
        patch("json.load", return_value="data") as mocked_json,
        patch("numpy.array") as mocked_numpy_array,
    ):
        input_handler.load_data()
        mocked_open.assert_called_once_with(mock_path)
        mocked_json.assert_called_once()
        mocked_numpy_array.assert_called_once()
