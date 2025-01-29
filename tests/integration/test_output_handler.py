import numpy as np

from pathlib import Path

from src.clustering.utils.output_handler import (
    NumpyOutputHandler,
    JSONOutputHandler,
)


def test_numpy_handler_save_to_file(mock_numpy_data: np.ndarray) -> None:
    """Test that `save_to_file` method of NumpyOutputHandler handles file
    saving correctly.

    Asserts:
        The numpy_handler is initialized without raising an error.
        Created file exists.
    """
    output_path = Path("clustered_data.npy")
    numpy_handler = NumpyOutputHandler()
    numpy_handler.save_to_file(mock_numpy_data)
    assert output_path.exists()


def test_json_handler_save_to_file(mock_json_data: np.ndarray) -> None:
    """Test that `save_to_file` method of JSONOutputHandler handles file
    saving correctly.

    Asserts:
        The numpy_handler is initialized without raising an error.
        Created file exists.
    """
    output_path = Path("clustered_data.json")
    numpy_handler = JSONOutputHandler()
    numpy_handler.save_to_file(mock_json_data)
    assert output_path.exists()
