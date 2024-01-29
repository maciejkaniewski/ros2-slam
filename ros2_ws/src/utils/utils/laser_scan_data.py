import array
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class LaserScanData:
    """
    Represents laser scan data with coordinates and measurements.
    """

    _coords: Tuple[float, float]
    _measurements: np.ndarray

    def __init__(self, coords: Tuple[float, float], measurements: array.array):
        """
        Initializes a LidarData object.

        Args:
            coords (Tuple[float, float]): The coordinates of the laser scan data.
            measurements (array.array): The measurements of the laser scan data.
        """
        self._coords = coords
        self._measurements = np.asarray(measurements, dtype=float)

    @property
    def coords(self) -> Tuple[float, float]:
        """
        Get the coordinates of the laser scan data.

        Returns:
            Tuple[float, float]: The coordinates of the laser scan data.
        """
        return self._coords

    @property
    def measurements(self) -> np.ndarray:
        """
        Get the measurements of the laser scan data.

        Returns:
            np.ndarray: The measurements of the laser scan data.
        """
        return self._measurements
