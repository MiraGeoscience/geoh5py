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

from typing import cast

import numpy as np

from geoh5py.shared.utils import get_unique_name_from_entities

from .data import Data
from .geometric_data import GeometricDataConstants
from .integer_data import IntegerData
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
    ):
        """
        Overwrite the copy method to ensure that the uid is set to None and
        transfer the GeometricDataValueMapType to the new entity type.
        """
        kwargs["uid"] = None

        new_data = super().copy(
            parent=parent,
            clear_cache=clear_cache,
            mask=mask,
            omit_list=["_metadata", "_data_maps"],
            **kwargs,
        )

        if not new_data:
            return None

        # Always overwrite the entity type name to protect the GeometricDataValueMapType
        new_data.entity_type.name = get_unique_name_from_entities(
            self.entity_type.name, self.workspace.types, types=Data
        )

        if self.data_maps is not None:
            data_maps = {}
            for child in self.data_maps.values():
                geometric_data = new_data.copy_data_map(child, clear_cache=clear_cache)
                if geometric_data:
                    data_maps[geometric_data.name] = geometric_data

            if data_maps:
                new_data.data_maps = data_maps

        return new_data

    def copy_data_map(
        self,
        data_map: GeometricDataConstants,
        name: str | None = None,
        clear_cache=True,
    ) -> GeometricDataConstants | None:
        """
        Create a copy with unique name of a GeometricDataConstant entity.

        :param data_map: An existing GeometricDataConstant to be copied.
        :param name: Name assigned to the data_map, or increments the original.
        :param clear_cache: Clear array attributes after copy.
        """
        name = get_unique_name_from_entities(
            name or data_map.name,
            self.parent.children,
            types=GeometricDataConstants,
        )
        data_type = self.parent.add_data_map_type(
            name, data_map.entity_type.value_map.map, self.entity_type.name
        )

        geometric_data = cast(
            GeometricDataConstants,
            self.workspace.copy_to_parent(
                data_map,
                self.parent,
                clear_cache=clear_cache,
                name=name,
            ),
        )

        # TODO: Clean up after GEOPY-2427
        if not geometric_data:
            return None

        geometric_data.entity_type = data_type
        return geometric_data

    @property
    def data_maps(self) -> dict[str, GeometricDataConstants] | None:
        """
        A reference dictionary mapping properties to numpy arrays.
        """
        if self._data_maps is None and self.metadata is not None:
            data_maps = {}
            for name, uid in self.metadata.items():
                child = self.workspace.get_entity(uid)[0]
                if isinstance(child, GeometricDataConstants):
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
                    not hasattr(val.entity_type, "value_map")
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
    def mapped_values(self) -> np.ndarray:
        """
        The values mapped from the reference data.
        """
        if self.value_map is None:
            raise ValueError("Entity type must have a value map.")

        return self.value_map.map_values(self.values)

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

    def add_data_map(self, name: str, values: np.ndarray | dict, public: bool = True):
        """
        Add a data map to the value map.

        :param name: The name of the data map.
        :param values: The data map to add.
        :param public: Whether the data map should be public.
        """
        data = self.parent.add_data_map(self, name, values, public)
        return data
