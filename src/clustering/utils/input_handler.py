import json
import numpy as np

from pathlib import Path


class InputHandler:
    """Class to handle load of the data from input file.

    This class provides data loading from .npy or .json file and transformation
    of the data to numpy array.

    Attributes:
        path (Path): Path to the input file.
    """

    def __init__(self, path: Path):
        """Initializes the InputHandler.

        Args:
            path (Path): Path to the input file.
        """
        self.path = path

    def load_data(self) -> np.ndarray:
        """Load input file and transform data into numpy array.

        This method checks existence of file and then load data from file
        based on suffix of the file.

        Returns:
            data (np.ndarray): The data to cluster.

        Raises:
            FileExistsError: If file at provided path does not exists.
            ValueError: If suffix of the file is not .npy or .json.
        """
        if not self.path.exists():
            raise FileExistsError("File does not exist.")
        if self.path.suffix == ".npy":
            return np.load(self.path)
        elif self.path.suffix == ".json":
            with open(self.path) as json_file:
                return np.array(json.load(json_file))
        else:
            raise ValueError(
                "Unsupported file format. Use .json or .yaml file."
            )
