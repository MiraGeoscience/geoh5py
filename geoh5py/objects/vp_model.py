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

# pylint: disable=too-many-instance-attributes, too-many-arguments

from __future__ import annotations

import uuid
from typing import cast

import numpy as np

from geoh5py.data import Data, FloatData, IntegerData, PrimitiveTypeEnum, ReferencedData
from geoh5py.objects import DrapeModel, Grid2D
from geoh5py.objects.grid_object import GridObject
from geoh5py.shared.utils import KEY_MAP, str2uuid, xy_rotation_matrix


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
    :param flag_property_id: UUID or name of the flag property.
    :param heterogeneous_property_id: UUID or name of the heterogeneous property.
    :param physical_data_name: Name of the physical data.
    :param unit_property_id: UUID or name of the unit property.
    :param weight_property_id: UUID or name of the weight property.
    """

    _VALUE_MAP = {
        100000: "VP_basement",
    }
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
        *,
        flag_property_id: str | uuid.UUID | np.ndarray = np.asarray([]),
        heterogeneous_property_id: str | uuid.UUID | np.ndarray = np.asarray([]),
        physical_data_name: str | np.ndarray = "Property",
        unit_property_id: str | uuid.UUID | np.ndarray = np.asarray([]),
        weight_property_id: str | uuid.UUID | np.ndarray = np.asarray([]),
        **kwargs,
    ):
        self._u_count: np.int32 = self.validate_count(u_count, "u")
        self._v_count: np.int32 = self.validate_count(v_count, "v")

        super().__init__(**kwargs)

        self.u_cell_size: float = u_cell_size
        self.v_cell_size: float = v_cell_size
        self.unit_property_id = self._promote_uuid_attribute(
            unit_property_id,
            "unit_property_id",
            PrimitiveTypeEnum.REFERENCED,
        )
        self.flag_property_id = self._promote_uuid_attribute(
            flag_property_id, "flag_property_id", PrimitiveTypeEnum.INTEGER
        )
        self.heterogeneous_property_id = self._promote_uuid_attribute(
            heterogeneous_property_id,
            "heterogeneous_property_id",
            PrimitiveTypeEnum.FLOAT,
        )
        self.physical_data_name = physical_data_name
        self.weight_property_id = self._promote_uuid_attribute(
            weight_property_id, "weight_property_id", PrimitiveTypeEnum.FLOAT
        )

    def _promote_uuid_attribute(
        self,
        value: str | uuid.UUID | np.ndarray | Data,
        name: str,
        primitive_type: PrimitiveTypeEnum,
    ) -> uuid.UUID:
        """
        Promote a string or UUID to a UUID.

        :param value: The value provided in the constructor.
        :param name: The name of the attribute validated.
        :param primitive_type: The primitive type of the attribute.
        """
        if isinstance(value, Data):
            if value not in self.children:
                raise ValueError(
                    f"Data '{value.name}' is not a child of the VPModel object."
                )
            return value.uid

        value = str2uuid(value)

        if isinstance(value, uuid.UUID):
            return value

        if not isinstance(value, np.ndarray):
            raise TypeError(
                f"Attribute '{name}' should be a 'uuid.UUID' or an array of values."
            )

        kwargs = {"association": "CELL", "type": primitive_type, "values": value}
        data = cast(
            Data,
            self.add_data(
                {KEY_MAP[name]: kwargs},
            ),
        )

        if isinstance(data, ReferencedData) and data.value_map is not None:
            value_map = data.value_map()
            value_map.update(self._VALUE_MAP)
            data.entity_type.value_map = data.entity_type.validate_value_map(value_map)

        return data.uid

    @property
    def flag_property_id(self) -> uuid.UUID:
        """
        VPmg flag property.
        """
        return self._flag_property_id

    @flag_property_id.setter
    def flag_property_id(self, value: uuid.UUID | IntegerData):
        if isinstance(value, IntegerData):
            value = value.uid

        if not isinstance(value, uuid.UUID):
            raise TypeError("Attribute 'flag_property_id' should be a 'uuid.UUID'.")
        self._flag_property_id = value

        if self.on_file:
            self.workspace.update_attribute(self, "attributes")

    @property
    def heterogeneous_property_id(self) -> uuid.UUID:
        """
        VPmg heterogenenous model id.
        """
        return self._heterogeneous_property_id

    @heterogeneous_property_id.setter
    def heterogeneous_property_id(self, value: uuid.UUID | FloatData):
        if isinstance(value, FloatData):
            value = value.uid

        if not isinstance(value, uuid.UUID):
            raise TypeError(
                "Attribute 'heterogeneous_property_id' should be a 'uuid.UUID'."
            )

        self._heterogeneous_property_id = value

        if self.on_file:
            self.workspace.update_attribute(self, "attributes")

    @property
    def physical_data_name(self) -> str:
        """
        VPmg physical property model data map name.
        """
        return self._physical_data_name

    @physical_data_name.setter
    def physical_data_name(self, value: str):
        if not isinstance(value, str):
            raise TypeError("Attribute 'physical_data_name' should be a 'str'")

        self._physical_data_name = value

        if self.on_file:
            self.workspace.update_attribute(self, "attributes")

    @property
    def unit_property_id(self) -> uuid.UUID | str | None:
        """
        VPmg unit model id.
        """
        return self._unit_property_id

    @unit_property_id.setter
    def unit_property_id(self, value: uuid.UUID | ReferencedData):
        if isinstance(value, ReferencedData):
            value = value.uid

        if not isinstance(value, uuid.UUID):
            raise TypeError("Attribute 'unit_property_id' should be a 'uuid.UUID'.")

        self._unit_property_id = value

        if self.on_file:
            self.workspace.update_attribute(self, "attributes")

    @property
    def weight_property_id(self) -> uuid.UUID | str | None:
        """
        VPmg weight model id.
        """
        return self._weight_property_id

    @weight_property_id.setter
    def weight_property_id(self, value: uuid.UUID | FloatData):
        if isinstance(value, FloatData):
            value = value.uid

        if not isinstance(value, uuid.UUID):
            raise TypeError(
                "Attribute 'weight_property_id' should be a 'uuid.UUID' or None"
            )
        self._weight_property_id = value

        if self.on_file:
            self.workspace.update_attribute(self, "attributes")

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

            v_grid, u_grid = np.meshgrid(
                np.cumsum(np.ones(self._v_count) * self.v_cell_size)
                - self.v_cell_size / 2,
                np.cumsum(np.ones(self._u_count) * self.u_cell_size)
                - self.u_cell_size / 2,
            )
            xyz = (
                rotation_matrix
                @ np.c_[np.ravel(u_grid), np.ravel(v_grid), self.prisms[:, 0]].T
            ).T
            xyz += self.origin[None, :]
            indices = (self.layers[:, 0] * self._v_count + self.layers[:, 1]).astype(
                int
            )
            centroids = xyz[indices, :]

            elevation = (
                self.prisms[:, 0].reshape((self.v_count, self.u_count)).T.flatten()
            )
            ind_colm = (
                self.prisms[:, 2].reshape((self.v_count, self.u_count)).T.flatten()
            )
            top_ind = np.r_[0, np.cumsum(ind_colm[:-1])].astype(int)
            for count in range(int(ind_colm.max()) - 1):
                mask = ind_colm > (count + 1)
                centroids[top_ind, 2] = np.mean(
                    [elevation, self.layers[top_ind, 2]], axis=0
                )
                elevation[mask] = self.layers[top_ind[mask], 2]
                top_ind[mask] += 1

            self._centroids = centroids

        return self._centroids

    @property
    def n_cells(self) -> int:
        """
        Total number of cells
        """
        return int(self.shape[2])

    @property
    def shape(self) -> np.ndarray:
        """
        Shape of the drape model in terms of number of prisms and layers.
        """
        return np.array([self._u_count, self._v_count, self.layers.shape[0]])

    @property
    def u_cell_size(self) -> float:
        """
        Cell size along the u-axis.
        """
        return self._u_cell_size

    @u_cell_size.setter
    def u_cell_size(self, value: float):
        self._u_cell_size = Grid2D.validate_cell_size(value, "u")
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
        self._v_cell_size = Grid2D.validate_cell_size(value, "v")
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

    def validate_metadata(self, value) -> dict | None:
        """
        Validate and format type of metadata value.
        """

        value = super().validate_metadata(value)

        if value is None:
            value = {}

        if value.get("VP", None) is None:
            value["VP"] = {"Base of model elevation (m)": self.layers[:, 2].min()}

        return value
