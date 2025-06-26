# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2025 Mira Geoscience Ltd.                                     '
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

import numpy as np

from geoh5py.shared.utils import str2uuid, xy_rotation_matrix

from .drape_model import DrapeModel
from .grid_object import GridObject


class VPModel(GridObject, DrapeModel):
    """
    VP Mesh object defined by a uniform grid in the u and v directions, and
    draped layers and prisms that represent the vertical structure of the model.

    Sub-class of :obj:`~geoh5py.objects.grid_object.GridObject` and
    :obj:`~geoh5py.objects.drape_model.DrapeModel`.

    :param u_cell_size: Cell size along the u-axis.
    :param u_count: Number of cells along the u-axis.
    :param v_cell_size: Cell size along the v-axis.
    :param v_count: Number of cells along the v-axis.
    """

    _TYPE_UID = uuid.UUID("{7d37f28f-f379-4006-984e-043db439ee95}")
    _LAYERS_DTYPE = np.dtype([("I", "<i4"), ("J", "<i4"), ("Bottom elevation", "<f8")])
    _PRISM_DTYPE = np.dtype(
        [
            ("Top elevation", "<f8"),
            ("First layer", "<i4"),
            ("Layer count", "<i4"),
        ]
    )
    _attribute_map = GridObject._attribute_map.copy()
    _attribute_map.update(
        {
            "Nu": "u_count",
            "Nv": "v_count",
            "Origin": "origin",
            "Rotation": "rotation",
            "U size": "u_cell_size",
            "V size": "v_cell_size",
            "Flag property ID": "flag_property_id",
            "Heterogeneous property ID": "heterogeneous_property_id",
            "Physical data name": "physical_data_name",
            "Unit property ID": "unit_property_id",
            "Weight property ID": "weight_property_id",
        }
    )

    def __init__(
        self,
        u_cell_size: float = 1.0,
        u_count: int = 1,
        v_cell_size: float = 1.0,
        v_count: int = 1,
        flag_property_id: str | uuid.UUID | None = None,
        heterogeneous_property_id: str | uuid.UUID | None = None,
        physical_data_name: str | None = None,
        unit_property_id: str | uuid.UUID | None = None,
        weight_property_id: str | uuid.UUID | None = None,
        **kwargs,
    ):
        self._u_count: np.int32 = self.validate_count(u_count, "u")
        self._v_count: np.int32 = self.validate_count(v_count, "v")

        super().__init__(**kwargs)

        self.u_cell_size: float = u_cell_size
        self.v_cell_size: float = v_cell_size
        self.flag_property_id = flag_property_id
        self.heterogeneous_property_id = heterogeneous_property_id
        self.physical_data_name = physical_data_name
        self.unit_property_id = unit_property_id
        self.weight_property_id = weight_property_id

    @property
    def flag_property_id(self) -> uuid.UUID | str | None:
        return self._flag_property_id

    @flag_property_id.setter
    def flag_property_id(self, value: uuid.UUID | str | None):
        value = str2uuid(value)

        if not isinstance(value, uuid.UUID | type(None)):
            raise TypeError("Attribute 'flag_property_id' should be a uuid or None")

        self._flag_property_id = value

    @property
    def heterogeneous_property_id(self) -> uuid.UUID | str | None:
        return self._heterogeneous_property_id

    @heterogeneous_property_id.setter
    def heterogeneous_property_id(self, value: uuid.UUID | str | None):
        value = str2uuid(value)

        if not isinstance(value, uuid.UUID | type(None)):
            raise TypeError(
                "Attribute 'heterogeneous_property_id' should be a uuid or None"
            )

        self._heterogeneous_property_id = value

    @property
    def physical_data_name(self) -> str | None:
        return self._physical_data_name

    @physical_data_name.setter
    def physical_data_name(self, value: str | None):
        if not isinstance(value, str | type(None)):
            raise TypeError("Attribute 'physical_data_name' should be a str or None")

        self._physical_data_name = value

    @property
    def unit_property_id(self) -> uuid.UUID | str | None:
        return self._unit_property_id

    @unit_property_id.setter
    def unit_property_id(self, value: uuid.UUID | str | None):
        value = str2uuid(value)

        if not isinstance(value, uuid.UUID | type(None)):
            raise TypeError("Attribute 'unit_property_id' should be a uuid or None")

        self._unit_property_id = value

    @property
    def weight_property_id(self) -> uuid.UUID | str | None:
        return self._weight_property_id

    @weight_property_id.setter
    def weight_property_id(self, value: uuid.UUID | str | None):
        value = str2uuid(value)

        if not isinstance(value, uuid.UUID | type(None)):
            raise TypeError("Attribute 'weight_property_id' should be a uuid or None")

        self._weight_property_id = value

    @property
    def centroids(self) -> np.ndarray:
        """
        Cell center locations in world coordinates, shape(*, 3).

        .. code-block:: python

            centroids = [
                [x_1, y_1, z_1],
                ...,
                [x_N, y_N, z_N]
            ]
        """
        if getattr(self, "_centroids", None) is None:
            rotation_matrix = xy_rotation_matrix(np.deg2rad(self.rotation))

            u_grid, v_grid = np.meshgrid(
                np.cumsum(np.ones(self._u_count) * self.u_cell_size),
                np.cumsum(np.ones(self._v_count) * self.v_cell_size),
            )
            xyz = (
                rotation_matrix
                @ np.c_[np.ravel(u_grid), np.ravel(v_grid), self.prisms[:, 0]].T
            ).T

            for ind, axis in enumerate(["x", "y", "z"]):
                xyz[:, ind] += self.origin[axis]

            indices = (self.layers[:, 0] * self._v_count + self.layers[:, 1]).astype(
                int
            )
            centroids = xyz[indices, :]

            elevation = self.prisms[:, 0]
            top_ind = np.r_[0, np.cumsum(self.prisms[:-1, 2])].astype(int)
            for count in range(int(self.prisms[:, 2].max()) - 1):
                mask = self.prisms[:, 2] > (count + 1)

                centroids[top_ind, 2] = np.mean(
                    [elevation, self.layers[top_ind, 2]], axis=0
                )
                elevation[mask] = self.layers[top_ind[mask], 2]
                top_ind[mask] += 1

            self._centroids = centroids

        return self._centroids

    @property
    def shape(self) -> np.ndarray:
        """
        Shape of the drape model in terms of number of prisms and layers.
        """
        return np.array([self._u_count, self._v_count, self.layers.shape[1]])

    @property
    def u_cell_size(self) -> float:
        """
        Cell size along the u-axis.
        """
        return self._u_cell_size

    @u_cell_size.setter
    def u_cell_size(self, value: float):
        self._u_cell_size = self.validate_size(value, "u")
        self._centroids = None

        if self.on_file:
            self.workspace.update_attribute(self, "attributes")

    @property
    def u_count(self) -> np.int32:
        """
        Number of cells along the u-axis.
        """
        return self._u_count

    @property
    def v_cell_size(self) -> float:
        """
        Cell size along the v-axis.
        """
        return self._v_cell_size

    @v_cell_size.setter
    def v_cell_size(self, value: float):
        self._v_cell_size = self.validate_size(value, "v")
        self._centroids = None

        if self.on_file:
            self.workspace.update_attribute(self, "attributes")

    @property
    def v_count(self) -> np.int32:
        """
        Number of cells along the v-axis.
        """
        return self._v_count

    @staticmethod
    def validate_count(value: int, axis: str) -> np.int32:
        """
        Validate and format type of count value.
        """
        if not isinstance(value, (np.integer, int)) or value < 1:
            raise TypeError(
                f"Attribute '{axis}_count' must be a type(int32) greater than 1."
            )

        return np.int32(value)

    @staticmethod
    def validate_size(value: float, axis: str) -> float:
        """
        Validate and format type of count value.
        """
        if not isinstance(value, float) or value <= 0:
            raise TypeError(
                f"Attribute '{axis}_count' must be a type(float) greater than 0."
            )

        return float(value)
