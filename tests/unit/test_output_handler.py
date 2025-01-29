import pytest
import numpy as np
import json

from typing import Union, Type
from unittest.mock import Mock, patch, mock_open

from src.clustering.utils.output_handler import (
    NumpyOutputHandler,
    JSONOutputHandler,
)

HandlerType = Union[Type[NumpyOutputHandler], Type[JSONOutputHandler]]


@pytest.mark.parametrize(
    "handler_class", [NumpyOutputHandler, JSONOutputHandler]
)
def test_output_handler_init(handler_class: HandlerType) -> None:
    """Test that handler_class is initialized correctly.

    Assert:
        The handler_class is initialized without raising an error.
    """
    handler_class()


def test_numpy_handler_save_to_file() -> None:
    """Test that `save_to_file` method of NumpyOutputHandler handles file
    saving correctly.

    Asserts:
        The numpy_handler is initialized without raising an error.
        Mocked_numpy_save is called once with correct arguments.
    """
    mock_data = Mock(spec=np.ndarray)
    numpy_handler = NumpyOutputHandler()
    with patch("numpy.save") as mocked_numpy_save:
        numpy_handler.save_to_file(mock_data)
        mocked_numpy_save.assert_called_once_with(
            "clustered_data.npy", mock_data
        )


def test_json_handler_save_to_file() -> None:
    """Test that `save_to_file` method of JSONOutputHandler handlers file
    saving correctly.

    Asserts:
        The json_handler is initialized without raising an error.
        The mocked_json_dumps is called once with correct arguments.
        The mocked_open is called once with correct arguments.
        The mocked_open().write is called once with correct arguments.
    """
    mock_data = Mock(spec=np.ndarray)
    mock_json = Mock(spec=json)
    json_handler = JSONOutputHandler()
    with (
        patch("json.dumps", return_value=mock_json) as mocked_json_dumps,
        patch("builtins.open", mock_open()) as mocked_open,
    ):
        json_handler.save_to_file(mock_data)
        mocked_json_dumps.assert_called_once_with(mock_data.tolist())
        mocked_open.assert_called_once_with("clustered_data.json", "w")
        mocked_open().write.assert_called_once_with(mock_json)
