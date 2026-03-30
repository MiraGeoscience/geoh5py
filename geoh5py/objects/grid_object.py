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

import uuid
from abc import ABC, abstractmethod
from numbers import Real

import numpy as np

from geoh5py.data import Data, DataAssociationEnum

from ..shared.utils import xy_rotation_matrix, yz_rotation_matrix
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
    def extent(self) -> np.ndarray:
        """
        Compute outer extent of mesh span in world coordinates.
        """
        u, v, w = np.meshgrid(self.span[0, :], self.span[1, :], self.span[2, :])
        xyz = self.uvw_to_xyz(np.c_[u.ravel(), v.ravel(), w.ravel()])
        return np.c_[xyz.min(axis=0), xyz.max(axis=0)].T

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

    @property
    @abstractmethod
    def span(self) -> np.ndarray:
        """
        Upper and lower limits along u, v and w directions.
        """

    def uvw_to_xyz(self, coordinates: np.ndarray) -> np.ndarray:
        """
        Apply rotation to the input coordinates.

        :param coordinates: Array of coordinates along the axes, transformed to world coordinates.

        :return:
        """

        rotation_matrix = xy_rotation_matrix(np.deg2rad(getattr(self, "rotation", 0)))
        dip_matrix = yz_rotation_matrix(np.deg2rad(getattr(self, "dip", 0)))

        xyz_dipped = dip_matrix @ coordinates.T
        xyz = (rotation_matrix @ xyz_dipped).T
        xyz += np.asarray(self._origin.tolist())[None, :]

        return xyz

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

    def _get_data_to_reshape(self, data: str | uuid.UUID | Data) -> Data:
        """
        Get a unique data entity with association 'CELL' from the data name, uid or object.

        :raises ValueError: if no data are found.
        :raises ValueError: if multiple data are found.
        :raises ValueError: if the data association is not 'CELL'.

        :param data: The data to get the values from.

        :return: The unique data.
        """
        if not isinstance(data, Data):
            data_list = self.get_data(data)

            if len(data_list) == 0:
                raise ValueError(f"No data '{data}' found.")
            if len(data_list) > 1:
                raise ValueError(
                    f"Multiple data '{data}' found. Please specify a unique data name or uid."
                )

            data = data_list[0]

        if data.association != DataAssociationEnum.CELL:
            raise ValueError(
                f"Data '{data.name}' has association '{data.association}'"
                ", expected 'CELL'."
            )
        return data
