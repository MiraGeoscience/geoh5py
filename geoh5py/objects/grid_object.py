# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2020-2026 Mira Geoscience Ltd.                                '
#                                                                              '
#  This file is part of geoh5py.                                               '
#                                                                              '
#  geoh5py is free software: you can redistribute it and/or modify             '
#  it under the terms of the GNU Lesser General Public License as published by '
#  the Free Software Foundation, either version 3 of the License, or           '
#  (at your option) any later version.                                         '
#                                                                              '
#  geoh5py is distributed in the hope that it will be useful,                  '
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              '
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               '
#  GNU Lesser General Public License for more details.                         '
#                                                                              '
#  You should have received a copy of the GNU Lesser General Public License    '
#  along with geoh5py.  If not, see <https://www.gnu.org/licenses/>.           '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


from __future__ import annotations

from abc import ABC, abstractmethod
from numbers import Real

import numpy as np

from .object_base import ObjectBase


ORIGIN_TYPE = np.dtype([("x", float), ("y", float), ("z", float)])


class GridObject(ObjectBase, ABC):
    """
    Base class for object with centroids.

    :param origin: Origin of the object.
    :param rotation: Rotation angle (clockwise) about the vertical axis.
    """

    _attribute_map = ObjectBase._attribute_map.copy()

    def __init__(
        self,
        origin: np.ndarray | tuple = (0.0, 0.0, 0.0),
        rotation: float = 0.0,
        **kwargs,
    ):
        self._centroids: np.ndarray | None = None

        super().__init__(**kwargs)

        self.origin = origin
        self.rotation = rotation

    @property
    @abstractmethod
    def centroids(self) -> np.ndarray:
        """
        Cell center locations in world coordinates of shape (n_cells, 3).
        """

    @property
    def n_cells(self) -> int:
        """
        Total number of cells
        """
        return int(np.prod(self.shape))

    @property
    def rotation(self) -> float:
        """
        Clockwise rotation angle (degree) about the vertical axis.
        """
        return self._rotation

    @rotation.setter
    def rotation(self, value: np.ndarray | Real):
        if isinstance(value, Real):
            value = np.r_[value]

        if not isinstance(value, np.ndarray) or value.shape != (1,):
            raise TypeError("Rotation angle must be a float of shape (1,)")

        self._centroids = None
        self._rotation = value.astype(float).item()

        if self.on_file:
            self.workspace.update_attribute(self, "attributes")

    @property
    def origin(self) -> np.ndarray:
        """
        Coordinates of the origin, shape (3, ).
        """
        return self._origin

    @origin.setter
    def origin(self, values: np.ndarray | list | tuple):
        if isinstance(values, (list, tuple)):
            values = np.array(values)

        if not isinstance(values, (np.ndarray, np.void)):
            raise TypeError(
                "Attribute 'origin' must be a list, tuple or numpy array. "
                f"Object of type {type(values)} provided."
            )

        if np.issubdtype(values.dtype, np.number):
            if len(values) != 3:
                raise ValueError(
                    "Attribute 'origin' must be a list or array of shape (3,). "
                    f"Array of shape {values.shape} provided."
                )

            values = np.asarray(tuple(values), dtype=ORIGIN_TYPE)

        if values.dtype != np.dtype(ORIGIN_TYPE):
            raise ValueError(f"Array of 'origin' must be of dtype = {ORIGIN_TYPE}")

        self._centroids = None
        self._origin = values

        if self.on_file:
            self.workspace.update_attribute(self, "attributes")

    @property
    @abstractmethod
    def shape(self) -> np.ndarray:
        """
        Cell center locations in world coordinates.
        """

    def validate_cell_mask(self, cell_mask: np.ndarray | None):
        """
        Validate cell mask array, which is the same as validate_mask for grid objects.

        :param cell_mask: Array of boolean values of shape (n_cells, ). If None provided,
        """
        return self.validate_mask(cell_mask)

    def validate_mask(self, mask: np.ndarray | None):
        """
        Validate mask array.

        :param mask: Array of boolean values of shape (n_cells, ). If None provided,
        """
        if mask is None:
            return None

        if (
            not isinstance(mask, np.ndarray)
            or mask.ndim != 1
            or mask.shape[0] != self.n_cells
            or mask.dtype != bool
        ):
            raise TypeError(
                "Mask must be a numpy array of shape (n_cells, ) and dtype 'bool'."
            )

        return mask
