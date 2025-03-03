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

from typing import cast

import numpy as np

from ..shared.utils import find_unique_name
from .data import Data
from .data_type import DataType, GeometricDataValueMapType, ReferenceDataType
from .geometric_data import GeometricDataConstants
from .integer_data import IntegerData
from .primitive_type_enum import PrimitiveTypeEnum
from .reference_value_map import ReferenceValueMap


class ReferencedData(IntegerData):
    """
    Reference data described by indices and associated strings.
    """

    def __init__(self, **kwargs):
        self._data_maps = None

        super().__init__(**kwargs)

    def copy(
        self,
        parent=None,
        *,
        clear_cache: bool = False,
        mask: np.ndarray | None = None,
        **kwargs,
    ) -> Data:
        """
        Overwrite the copy method to ensure that the uid is set to None and
        transfer the GeometricDataValueMapType to the new entity type.
        """
        kwargs["uid"] = None

        new_data = cast(
            ReferencedData,
            super().copy(
                parent=parent,
                clear_cache=clear_cache,
                mask=mask,
                omit_list=["_metadata", "_data_maps"],
                **kwargs,
            ),
        )

        # Always overwrite the entity type name to protect the GeometricDataValueMapType
        new_data.entity_type.name = find_unique_name(
            self.entity_type.name,
            [tp.name for tp in self.workspace.types if isinstance(tp, DataType)],
        )

        if self.data_maps is not None:
            for name, child in self.data_maps.items():
                new_data.add_data_map(name, child.entity_type.value_map.map)

        return new_data

    @property
    def data_maps(self) -> dict[str, GeometricDataConstants] | None:
        """
        A reference dictionary mapping properties to numpy arrays.
        """
        if self._data_maps is None and self.metadata is not None:
            data_maps = {}
            for name, uid in self.metadata.items():
                child = self.workspace.get_entity(uid)[0]
                if isinstance(child, GeometricDataConstants) and isinstance(
                    child.entity_type, GeometricDataValueMapType
                ):
                    data_maps[name] = child

            if data_maps:
                self._data_maps = data_maps

        return self._data_maps

    @data_maps.setter
    def data_maps(self, data_map: dict[str, GeometricDataConstants] | None):
        if data_map is not None:
            if not isinstance(data_map, dict):
                raise TypeError("Property maps must be a dictionary")
            for key, val in data_map.items():
                if not isinstance(val, GeometricDataConstants):
                    raise TypeError(
                        f"Property maps value for '{key}' must be a 'GeometricDataConstants'."
                    )

                if (
                    not isinstance(val.entity_type, GeometricDataValueMapType)
                    or val.entity_type.value_map is None
                ):
                    raise ValueError(
                        f"Property maps value for '{key}' must have a "
                        "'GeometricDataValueMapType' entity type and a 'value_map' assigned."
                    )

            self.metadata = {child.name: child.uid for child in data_map.values()}

        self._data_maps = data_map

        if self.on_file:
            self.workspace.update_attribute(self, "data_map")

    @property
    def entity_type(self) -> ReferenceDataType:
        """
        The associated reference data type.
        """
        return self._entity_type

    @entity_type.setter
    def entity_type(self, data_type: ReferenceDataType):
        if not isinstance(data_type, ReferenceDataType):
            raise TypeError("entity_type must be of type ReferenceDataType")

        self._entity_type = data_type

        if self.on_file:
            self.workspace.update_attribute(self, "entity_type")

    @property
    def mapped_values(self) -> np.ndarray:
        """
        The values mapped from the reference data.
        """
        if self.value_map is None:
            raise ValueError("Entity type must have a value map.")

        return self.value_map.map_values(self.values)

    @classmethod
    def primitive_type(cls) -> PrimitiveTypeEnum:
        return PrimitiveTypeEnum.REFERENCED

    @property
    def value_map(self) -> ReferenceValueMap | None:
        """
        Pointer to the :obj:`data.data_type.DataType.value_map`
        """
        return self.entity_type.value_map

    def remove_data_map(self, name: str):
        """
        Remove a data map from the list of children.

        :param name: The name of the data map to remove.
        """

        if self.data_maps is None or name not in self.data_maps:
            return

        child = self.data_maps[name]
        child.allow_delete = True
        self.workspace.remove_entity(child)
        self.workspace.remove_entity(child.entity_type)

        del self.data_maps[name]
        self.data_maps = self._data_maps

    def add_data_map(self, name: str, data: np.ndarray | dict):
        """
        Add a data map to the value map.

        :param name: The name of the data map.
        :param data: The data map to add.
        """
        value_map = self.data_maps or {}

        if name in value_map:
            raise KeyError(f"Data map '{name}' already exists.")

        if not isinstance(data, dict | np.ndarray):
            raise TypeError("Data map values must be a numpy array or dict")

        if self.entity_type.value_map is None:
            raise ValueError("Entity type must have a value map.")

        reference_data = ReferenceValueMap(data, name=name)
        # TODO: Enforce that the keys of the data map are a subset
        #  of the value map keys once GA changes its behavior
        # if not set(reference_data.map["Key"]).issubset(
        #     set(self.entity_type.value_map.map["Key"])
        # ):
        #     raise KeyError("Data map keys must be a subset of the value map keys.")
        #

        data_type = GeometricDataValueMapType(
            self.workspace,
            value_map=reference_data,
            parent=self.parent,
            name=self.entity_type.name + f": {name}",
        )
        geom_data = self.parent.add_data(
            {
                name: {
                    "association": self.association,
                    "entity_type": data_type,
                }
            }
        )
        value_map[name] = geom_data
        self.data_maps = value_map
