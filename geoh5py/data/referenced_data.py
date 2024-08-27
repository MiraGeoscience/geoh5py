#  Copyright (c) 2024 Mira Geoscience Ltd.
#
#  This file is part of geoh5py.
#
#  geoh5py is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  geoh5py is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with geoh5py.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import numpy as np

from .data_type import ReferenceDataType
from .integer_data import IntegerData
from .primitive_type_enum import PrimitiveTypeEnum
from .reference_value_map import ReferenceValueMap


class ReferencedData(IntegerData):
    """
    Reference data described by indices and associated strings.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def data_maps(self):
        """
        Pointer to the :obj:`data.data_type.DataType.value_map`
        """
        return self.entity_type.value_map.data_maps

    @property
    def entity_type(self) -> ReferenceDataType:
        """
        :obj:`~geoh5py.data.data_type.ReferenceDataType`
        """
        return self._entity_type

    @entity_type.setter
    def entity_type(self, data_type: ReferenceDataType):

        if not isinstance(data_type, ReferenceDataType):
            raise TypeError("entity_type must be of type ReferenceDataType")

        self._entity_type = data_type

        if self.on_file:
            self.workspace.update_attribute(self, "entity_type")

    @classmethod
    def primitive_type(cls) -> PrimitiveTypeEnum:
        return PrimitiveTypeEnum.REFERENCED

    @property
    def value_map(self) -> ReferenceValueMap:
        """
        Pointer to the :obj:`data.data_type.DataType.value_map`
        """
        return self.entity_type.value_map

    def add_data_map(self, name: str, data: np.ndarray | dict):
        """
        Add a data map to the value map.

        :param name: The name of the data map.
        :param data: The data map to add.
        """
        value_map = self.entity_type.value_map.data_maps or {}

        if name in value_map:
            raise KeyError(f"Data map '{name}' already exists.")

        if not isinstance(data, dict | np.ndarray):
            raise TypeError("Data map values must be a numpy array or dict")

        if isinstance(data, np.ndarray) and data.ndim != 2:
            raise ValueError("Data map must be a 2D array")

        data = np.array(list(dict(data)), dtype=ReferenceValueMap.MAP_DTYPE)

        if not set(data["Key"]).issubset(set(self.entity_type.value_map.map["Key"])):
            raise KeyError("Data map keys must be a subset of the value map keys.")

        value_map[name] = data

        self.entity_type.value_map.data_maps = value_map

        if self.on_file:
            self.workspace.update_attribute(self.entity_type, "value_map")
