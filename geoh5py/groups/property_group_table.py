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
from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np

from ..data import DataAssociationEnum
from ..data.primitive_type_enum import DataTypeEnum


if TYPE_CHECKING:  # pragma: no cover
    from .property_group import PropertyGroup


class PropertyGroupTable(ABC):
    """
    A class to store the information of a property group.

    :param property_group: The property group to extract the data from.
    """

    locations_keys = ("X", "Y", "Z")

    def __init__(self, property_group: PropertyGroup):
        if not hasattr(property_group, "property_group_type"):
            raise TypeError("'property_group' must be a PropertyGroup object.")

        if hasattr(property_group.parent, "collar"):
            raise NotImplementedError(
                "PropertyGroupTable is not supported for Drillhole objects."
            )

        self._property_group: PropertyGroup = property_group

    def __call__(
        self, spatial_index: bool = False, use_uids: bool = False
    ) -> np.ndarray | None:
        """
        Create a structured array with the data of the properties.

        :param spatial_index: If True, the spatial index is added to the table.
        :param use_uids: If True, the uids are used as columns name.

        :return: A table with the data of the properties.
        """
        if (
            self.property_group.properties is None
            or self.property_group.properties_name is None
        ):
            return None

        keys: list[UUID] = self.property_group.properties
        names: list[str] | list[UUID] = (
            self.property_group.properties
            if use_uids
            else self.property_group.properties_name
        )

        output_array = self._create_empty_structured_array(names, keys, spatial_index)

        if spatial_index:
            for idx, key in enumerate(self.locations_keys):
                output_array[key] = self.locations[:, idx]

        for key, name in zip(keys, names, strict=False):  # type: ignore
            data = self.property_group.parent.get_data(key)[0]
            output_array[str(name)] = data.values

        return output_array

    def _create_empty_structured_array(
        self,
        properties_name: list[str] | list[UUID],
        properties_keys: list[UUID],
        spatial_index: bool = False,
    ) -> np.ndarray:
        """
        Create an empty structured array that can contains the data.

        :param properties_name: The names of the properties.
        :param properties_keys: The keys of the properties.
        :param spatial_index: If True, the association is added to the table.

        :return: an empty structured array.
        """
        dtypes = (
            [(loc, np.float32) for loc in self.locations_keys] if spatial_index else []
        )

        for key, name in zip(properties_keys, properties_name, strict=False):
            data = self.property_group.parent.get_data(key)[0]
            dtype: type | str = DataTypeEnum.from_primitive_type(
                data.entity_type.primitive_type
            )
            if dtype not in [np.float32, np.int32, np.uint32, bool]:
                dtype = "O"
            dtypes.append((str(name), dtype))

        empty_array = np.recarray((self.size,), dtype=dtypes)

        return empty_array

    @property
    def locations(self) -> np.ndarray:
        """
        The locations of the association table.

        This function is needed as a data can be both associated to cell or
        vertex in CellObjects.
        """
        if self.property_group.association == DataAssociationEnum.VERTEX:
            return self.property_group.parent.vertices  # type: ignore

        if self.property_group.association == DataAssociationEnum.CELL:
            return self.property_group.parent.centroids  # type: ignore

        raise ValueError(
            f"The association {self.property_group.association} is not supported. "
            f"Only VERTEX and CELL associations are supported."
        )

    @property
    def property_group(self) -> PropertyGroup:
        """
        The property group to extract the data from.
        """
        return self._property_group

    @property
    def size(self) -> int:
        """
        The size of the properties in the group.
        """
        return self.locations.shape[0]
