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

from abc import ABC
from typing import TYPE_CHECKING, Any
from uuid import UUID

import numpy as np

from ..data import DataAssociationEnum
from ..data.primitive_type_enum import DataTypeEnum
from ..shared.utils import str2uuid, to_tuple


if TYPE_CHECKING:  # pragma: no cover
    from .property_group import PropertyGroup


class PropertyGroupTable(ABC):
    """
    A class to store the information of a property group.

    :param property_group: The property group to extract the data from.
    """

    locations_keys = ("X", "Y", "Z")

    def __init__(self, property_group: PropertyGroup):
        self._properties_type: dict | None = None
        self._size: int | None = None

        self._property_group: PropertyGroup = property_group

    def _convert_names_to_uid(self, names: tuple[str | UUID]) -> tuple:
        """
        Convert the names of the properties to their UUID.

        :param names: The names of the properties to find back.

        :return: A list of UUID.
        """

        uuids: tuple = ()

        for name in names:
            uid = str2uuid(name)

            if isinstance(uid, str):
                data = self.property_group.parent.get_data(uid)
                if len(data) == 0:
                    raise ValueError(f"Data with name {name} not found.")
                if len(data) > 1:
                    raise ValueError(f"Multiple data with name {name} found.")
                uid = data[0].uid

            if (
                self.property_group.properties is None
                or not isinstance(uid, UUID)
                or uid not in self.property_group.properties
            ):
                raise ValueError(f"Data '{name}' not found in the property group.")

            uuids += (uid,)

        if len(uuids) == 0:
            raise ValueError(f"No data found for '{names}' in the property group.")

        return uuids

    def _create_empty_structured_array(
        self,
        keys: tuple[UUID],
        names: tuple[Any],
        spatial_index: bool = False,
    ) -> np.ndarray:
        """
        Create an empty structured array that can contains the data.

        :param keys: The list containing the data.
        :param spatial_index: If True, the association is added to the table.

        :return: an empty structured array.
        """
        if self.size is None or self.properties_type is None:
            raise ValueError("The PropertyGroup has no property.")

        dtypes = (
            [(loc, np.float32) for loc in self.locations_keys] if spatial_index else []
        )
        no_data_values = [np.nan] * 3 if spatial_index else []

        for key, name in zip(keys, names, strict=False):
            dtype, nan_value = self.properties_type[key]
            dtypes.append((str(name), dtype))
            no_data_values.append(nan_value)

        empty_array = np.recarray((self.size,), dtype=dtypes)

        for name, ndv in zip(empty_array.dtype.names, no_data_values, strict=False):
            empty_array[name].fill(ndv)

        return empty_array

    @property
    def association_columns(self) -> str:
        """
        The columns of the association table.

        This function is needed as a data can be both associated to cell or
        vertex in CellObjects.
        """
        if self.property_group.association == DataAssociationEnum.VERTEX:
            return "vertices"
        if self.property_group.association == DataAssociationEnum.CELL:
            return "centroids"
        return "locations"

    @property
    def property_group(self) -> PropertyGroup:
        """
        The property group to extract the data from.
        """
        return self._property_group

    @property
    def property_table(self) -> np.ndarray | None:
        """
        Create a structured array with the data of the properties.

        This structured array also contains the spatial index.

        :return: The table with the data of the properties.
        """
        if self.property_group.properties is None:
            return None

        return self.property_table_by_name(
            self.property_group.properties,  # type: ignore
            spatial_index=True,
        )

    def property_table_by_name(
        self, names: list[str | UUID] | str | UUID, spatial_index: bool = False
    ) -> np.ndarray:
        """
        Create a structured array with the data of the properties.

        :param names: The names of the properties to extract.
        :param spatial_index: If True, the spatial index is added to the table.

        :return: A table with the data of the properties.
        """
        names_ = to_tuple(names)

        keys = self._convert_names_to_uid(names_)

        output_array = self._create_empty_structured_array(keys, names_, spatial_index)

        if spatial_index:
            for idx, key in enumerate(self.locations_keys):
                output_array[key] = getattr(
                    self.property_group.parent, self.association_columns
                )[:, idx]

        for key, name in zip(keys, names_, strict=False):
            data = self.property_group.parent.get_data(key)[0]
            output_array[str(name)] = data.values

        return output_array

    @property
    def properties_type(self) -> dict | None:
        """
        The types of the properties in the group.
        """
        if self._properties_type is None:
            self.update()

        return self._properties_type

    @property
    def size(self) -> int | None:
        """
        The size of the properties in the group.
        """
        if self._size is None:
            self.update()

        return self._size

    def update(self):
        """
        Find the dtypes for all properties in the group.
        Also check the length of all data.
        """
        properties_type: dict = {}
        sizes: list = []

        if self._property_group.properties is None:
            return

        for property_ in self._property_group.properties:
            data = self.property_group.parent.get_data(property_)[0]

            dtype = DataTypeEnum.from_primitive_type(data.entity_type.primitive_type)
            if dtype not in [np.float32, np.int32, np.uint32, bool]:
                dtype = "O"

            properties_type[property_] = (dtype, data.nan_value)
            sizes.append(data.values.size)

        if not all(size == sizes[0] for size in sizes):
            raise ValueError("All properties must have the same length.")

        self._size = sizes[0]
        self._properties_type = properties_type
