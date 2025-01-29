import numpy as np
import json

from abc import ABC, abstractmethod


class OutputHandler(ABC):
    """Abstract class for handling output."""

    @abstractmethod
    def save_to_file(self, data: np.ndarray) -> None:
        """Saves data to corresponding file format.

        This method must be implemented by subclasses.

        Args:
            data(np.ndarray): Data to be stored into a file.
        """
        pass


class NumpyOutputHandler(OutputHandler):
    """Implementation of OutputHandler for numpy file format."""

    def save_to_file(self, data: np.ndarray) -> None:
        """Saves data to numpy file format."""
        np.save("clustered_data.npy", data)


class JSONOutputHandler(OutputHandler):
    """Implementation of OutputHandler for json file format."""

    def save_to_file(self, data: np.ndarray) -> None:
        """Transform data into json and saves it to json file format."""
        json_data = json.dumps(data.tolist())
        with open("clustered_data.json", "w") as json_file:
            json_file.write(json_data)
