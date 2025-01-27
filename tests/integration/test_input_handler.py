import pytest
import numpy as np

from pathlib import Path

from src.clustering.utils.input_handler import InputHandler


@pytest.mark.parametrize(
    "file_path", ["numpy_path", "json_path"], indirect=True
)
def test_input_handler_init(file_path: Path) -> None:
    """Test that the InputHandler is initialized correctly.

    Args:
        file_path (Path): The path to the input data file.

    Asserts:
        The InputHandler is initialized without raising an error.
        The `path` attribute is of type Path.
    """
    input_handler = InputHandler(path=file_path)
    assert isinstance(input_handler.path, Path)


def test_load_data_invalid_suffix(text_path: Path) -> None:
    """Test that `load_data` method handles invalid suffix correctly.

    Args:
        text_path (Path): Path to text file.

    Asserts:
        The InputHandler is initialized without raising an error.
        The ValueError is raised.
    """
    input_handler = InputHandler(path=text_path)
    with pytest.raises(ValueError):
        input_handler.load_data()


@pytest.mark.parametrize(
    "file_path", ["numpy_path", "json_path"], indirect=True
)
def test_load_data_valid_file(file_path: Path, mock_data: np.ndarray) -> None:
    """Test that `load_data` method handles valid suffix correctly.

    Args:
        file_path (Path): Path to file with valid suffix.
        mock_data (np.ndarray): Mock np.ndarray data.

    Asserts:
        The loaded data is of np.ndarray type.
        The loaded data is matching the mock data.
    """
    input_handler = InputHandler(path=file_path)
    loaded_data = input_handler.load_data()
    assert isinstance(loaded_data, np.ndarray)
    np.testing.assert_array_equal(loaded_data, mock_data)
