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

from abc import ABC
from typing import TYPE_CHECKING, Any
from uuid import UUID

import numpy as np

from ...data.data_type import DataType, ReferencedValueMapType
from ..utils import decode_byte_array, find_unique_name, str2uuid, to_tuple
from .property_group import ConcatenatedPropertyGroup


if TYPE_CHECKING:  # pragma: no cover
    from .concatenator import Concatenator
    from .data import ConcatenatedData
    from .drillhole import ConcatenatedDrillhole


class DrillholesGroupTable(ABC):
    """
    A class to store the information of a PropertyGroup.

    :param parent: The parent DrillholesConcatenator.
    :param name: The name of the PropertyGroup to extract.
    """

    def __init__(self, parent: Concatenator, name: str):
        self._association: tuple | None = None
        self._index_by_drillhole: dict | None = None
        self._properties: dict[str, DataType] | None = None

        self._property_groups: dict[UUID, ConcatenatedPropertyGroup] = (
            self._get_property_groups(parent, name)
        )
        self._parent: Concatenator = parent
        self._name: str = name

    def _create_empty_structured_array(
        self,
        names: tuple[str],
        drillhole: bool = True,
        mapped: bool = False,
    ) -> np.ndarray:
        """
        Create an empty structured array that can contains the data.

        :param names: The names to extract.
        :param drillhole: If True, the drillholes are added to the table.
        :param mapped: Map the referenced data back to the its descriptions instead of indexes.

        :return: an empty structured array.
        """
        dtypes = [("Drillhole", "O")] if drillhole else []
        no_data_values = [None] if drillhole else []

        for name in self.association:
            if name in names:
                dtypes.append((name, np.float32))
                no_data_values.append(np.nan)

        for name, data_type in self.properties_type.items():
            if name in names:
                if (data_type.dtype not in [np.float32, np.int32, np.uint32, bool]) or (
                    isinstance(data_type, ReferencedValueMapType) and mapped
                ):
                    dtype = "O"
                else:
                    dtype = data_type.dtype
                dtypes.append((name, dtype))
                no_data_values.append(self.nan_value_from_name(name))

        empty_array = np.recarray(
            (self.parent.data[self.association[0]].shape[0],), dtype=dtypes
        )

        for name, ndv in zip(empty_array.dtype.names, no_data_values, strict=False):
            empty_array[name].fill(ndv)

        return empty_array

    def _depth_table_by_key(
        self,
        names: tuple,
        drillholes: bool = True,
        mapped: bool = False,
    ) -> np.ndarray:
        """
        Get a table with all the data associated with depth for every drillhole object.

        The Drillhole name is added at the beginning of the table for every row.
        The table is based on the association and contains nan values if no data is found.

        :param names: The names to extract.
        :param drillholes: If True, the drillholes are added to the table.
        :param mapped: Map the referenced data back.

        :return: a structured array with all the data.
        """
        if self.index_by_drillhole is None:
            raise ValueError("No drillhole found in the concatenator.")

        output_array = self._create_empty_structured_array(
            names=names, drillhole=drillholes, mapped=mapped
        )
        current_index = 0

        for object_, data_dict in self.index_by_drillhole.items():
            for name, info in data_dict.items():
                if name in names:
                    temp_array = self.parent.data[name][info[0] : info[0] + info[1]]
                    output_array[name][
                        current_index : current_index + temp_array.shape[0]
                    ] = temp_array

            if drillholes:
                output_array["Drillhole"][
                    current_index : current_index + data_dict[self.association[0]][1]
                ] = object_

            current_index += data_dict[self.association[0]][1]

        if mapped:
            return self._replace_referenced_data(output_array)

        return output_array

    @staticmethod
    def _get_property_groups(parent, name) -> dict[UUID, ConcatenatedPropertyGroup]:
        """
        Get all the property groups with a given name in the concatenator.

        :param parent: the concatenator object.
        :param name: the name to get the property groups for.

        :return: a  dictionary of property groups with drillhole object as Key.
        """
        if not hasattr(parent, "property_group_ids"):
            raise TypeError("The parent must be a Concatenator instance.")

        property_groups: dict[UUID, ConcatenatedPropertyGroup] = {}
        for property_group_uid in parent.property_group_ids:
            property_group = parent.workspace.get_entity(str2uuid(property_group_uid))[
                0
            ]
            if (
                isinstance(property_group, ConcatenatedPropertyGroup)
                and property_group.name == name
            ):
                property_groups[property_group.parent.uid] = property_group

        if not property_groups:
            raise ValueError(
                f"No property group with name '{name}' found in the concatenator."
            )

        return property_groups

    def _get_properties_names_types(self):
        if not self._properties:
            properties: dict[str, DataType] = {}

            for property_group in self.property_groups.values():
                if property_group.properties is None:
                    continue

                for property_ in property_group.properties:
                    property_temp = property_group.parent.get_data(property_)[0]

                    if (
                        property_temp.name not in self.association
                        and property_temp.name not in properties
                    ):
                        properties[property_temp.name] = property_temp.entity_type

            # sort the properties names and dtypes

            if properties:
                self._properties = dict(sorted(properties.items()))

    def _replace_referenced_data(self, output_array: np.ndarray) -> np.ndarray:
        """
        Replace the referenced data in the output array.

        :param output_array: The array to replace the data in.

        :return: The array with the replaced data.
        """
        # get the
        for name in output_array.dtype.names:
            if name in self.properties_type and isinstance(
                self.properties_type[name], ReferencedValueMapType
            ):
                output_array[name] = decode_byte_array(
                    self.properties_type[name].value_map.map_values(output_array[name]),
                    str,
                )

        return output_array

    def add_values_to_property_group(
        self, name: str, values: np.ndarray, data_type: DataType | None = None
    ):
        """
        Push the values to each drillhole of the property group based on association.

        :param name: The name of the data to push.
        :param values: The values to push.
        :param data_type: The data type associated to description;
            useful especially for referenced data.
        """
        name = find_unique_name(name, list(self.parent.data.keys()))

        if not isinstance(name, str) or name in self.parent.data:
            raise KeyError("The name must be a string not present in data.")

        # ensure the length of the values is the same as the length of the template
        if values.shape != self.parent.data[self.association[0]].shape:
            raise ValueError(
                "The length of the values must be the same as the association "
                f"({self.parent.data[self.association[0]].shape})."
            )

        if not isinstance(data_type, DataType):
            primitive_type = DataType.primitive_type_from_values(values)
            data_type = DataType.find_or_create_type(
                self.parent.workspace, primitive_type, name=name
            )

        for drillhole_uid, indices in self.index_by_drillhole.items():
            # get the drillhole
            drillhole: ConcatenatedDrillhole = self.parent.workspace.get_entity(  # type: ignore
                str2uuid(drillhole_uid)
            )[0]

            # add data to the drillhole
            drillhole.add_data(
                {
                    name: {
                        **{
                            "values": values[
                                indices[self.association[0]][0] : indices[
                                    self.association[0]
                                ][0]
                                + indices[self.association[0]][1]
                            ],
                        },
                        "entity_type": data_type,
                    },
                },
                property_group=self.property_groups[drillhole.uid],
            )

        self._update_drillholes_group_table(name, data_type)

    @property
    def association(self) -> tuple:
        """
        The depth association of the PropertyGroup.
        """
        if self._association is None:
            if self.property_group_type == "Interval table":
                self._association = (
                    list(self.property_groups.values())[0].from_.name,
                    list(self.property_groups.values())[0].to_.name,
                )
            elif self.property_group_type == "Depth table":
                self._association = (
                    list(self.property_groups.values())[0].depth_.name,
                )
            else:
                raise TypeError(
                    f"The property group type '{self.property_group_type}' is not supported."
                )

        return self._association

    @property
    def depth_table(
        self,
    ) -> np.ndarray:
        """
        Get a table with all the data associated with depth for every drillhole object.

        The Drillhole name is added at the beginning of the table for every row.
        The table is based on the association and contains nan values if no data is found.

        :return: a structured array with all the data.
        """
        # get the depth table
        return self._depth_table_by_key(self.association + self.properties, True)

    def depth_table_by_name(
        self, names: tuple[str] | str, spatial_index: bool = False, mapped: bool = False
    ) -> np.ndarray:
        """
        Get a table with specific data associated with depth for every drillhole object.

        :param names: The names to extract.
        :param spatial_index: If True, the spatial index is added to the table.
        :param mapped: Map the referenced data back.

        :return: a table containing the Drillholes, the association and the data.
        """
        names = to_tuple(names)

        # ensure names are in properties
        if not all(name in self.properties for name in names):
            raise KeyError("The names are not in the list of properties.")

        if spatial_index:
            return self._depth_table_by_key(self.association + names, True, mapped)

        return self._depth_table_by_key(names, False, mapped)

    @property
    def index_by_drillhole(
        self,
    ) -> dict[bytes, dict[str, list[int]]]:
        """
        Get for every object index and count of all the data in 'association' and 'properties'

        :return: A dictionary with the object uuid and the index of all the data.
        """
        if self._index_by_drillhole is None:
            index_by_drillhole: dict[bytes, dict[str, list[int]]] = {}
            names = self.association + self.properties
            for drillhole in np.sort(self._parent.index[names[0]], order="Start index")[
                "Object ID"
            ]:
                index_by_drillhole[drillhole] = {}
                for name in names:
                    if drillhole in self._parent.index[name]["Object ID"]:
                        index_by_drillhole[drillhole][name] = list(
                            self._parent.index[name][
                                self._parent.index[name]["Object ID"] == drillhole
                            ][0]
                        )[:2]
                    else:
                        index_by_drillhole[drillhole][name] = [0, 0]

            if index_by_drillhole:
                self._index_by_drillhole = index_by_drillhole
            else:
                raise ValueError("No drillhole found in the concatenator.")

        return self._index_by_drillhole

    @property
    def name(self) -> str:
        """
        The name of the PropertyGroup.
        """
        return self._name

    def nan_value_from_name(self, name: str) -> Any:
        """
        Get the nan value of a data from the name.

        :param name: The name of the data to get.

        :return: The nan value.
        """
        if name not in self.properties + self.association:
            raise KeyError(f"The name '{name}' is not in the list of properties.")

        # get the data of the know association
        data: ConcatenatedData = self.parent.workspace.get_entity(  # type: ignore
            str2uuid(self.parent.index[name][0][-2])
        )[0].get_data(str2uuid(self.parent.index[name][0][-1]))[0]

        return data.nan_value

    @property
    def parent(self) -> Concatenator:
        """
        The parent Concatenator object.
        """
        return self._parent

    @property
    def properties(self) -> tuple:
        """
        The names of the associated data.
        """
        if not self._properties:
            self._get_properties_names_types()

        if self._properties is None:
            return ()

        return tuple(self._properties.keys())

    @property
    def properties_type(self) -> dict:
        """
        A mapper of the type in function of the properties name
        """
        if not self._properties:
            self._get_properties_names_types()

        if self._properties is None:
            return {}

        return self._properties

    @property
    def property_group_type(self) -> str:
        """
        The type of the PropertyGroup.
        """
        return list(self.property_groups.values())[0].property_group_type

    @property
    def property_groups(self) -> dict[UUID, ConcatenatedPropertyGroup]:
        """
        Get all the property groups in the concatenator.

        :return: A list containing all the property groups.
        """
        return self._property_groups

    def _update_drillholes_group_table(self, name: str, data_type: DataType):
        """
        Update the drillholes group table with a new property group.

        :param name: The name of the property group to update.
        :param data_type: The data type of the property group.
        """
        self.parent.update_data_index()
        self.parent.workspace.update_attribute(self.parent, "concatenated_attributes")
        self._property_groups = self._get_property_groups(self.parent, self.name)

        if self.properties and self._properties is not None:
            self._properties[name] = data_type
        else:
            self._properties = {name: data_type}

        if self._index_by_drillhole is not None:
            for value in self._index_by_drillhole.values():
                value[name] = value[self.association[0]]
