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

from ..utils import str2uuid, to_tuple
from .property_group import ConcatenatedPropertyGroup

if TYPE_CHECKING:
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
        self._properties: tuple | None = None

        self._property_groups: dict[UUID, ConcatenatedPropertyGroup] = (
            self._get_property_groups(parent, name)
        )
        self._parent: Concatenator = parent
        self._name: str = name

    @staticmethod
    def _create_structured_array(output: np.ndarray, names: tuple[str]) -> np.ndarray:
        """
        Create a structured array from the output of the function get_depth_table.

        :param output: The data to pass to structured array.
        :param names: The name of the columns of the structured array.

        :return: The structured array.
        """
        # create the structured array
        dtype = []
        for idx, data_name in enumerate(names):
            type_temp = np.array([output[0, idx]]).dtype
            if type_temp.kind in ["S", "U"]:
                dtype.append((data_name, "O"))
            else:
                dtype.append((data_name, type_temp))

        return np.core.records.fromarrays(output.T, dtype=dtype)

    def _depth_table_by_key(
        self,
        names: tuple,
        drillholes: bool = True,
    ) -> np.ndarray:
        """
        Get a table with all the data associated with depth for every drillhole object.

        The Drillhole name is added at the beginning of the table for every row.
        The table is based on the association and contains nan values if no data is found.

        :param names: The names to extract.
        :param drillholes: If True, the drillholes are added to the table.

        :return: a structured array with all the data.
        """
        if self.index_by_drillhole is None:
            raise ValueError("No drillhole found in the concatenator.")

        all_data_list = []
        for object_, data_dict in self.index_by_drillhole.items():
            data_list: list = []
            no_data_values: list = []
            for name, info in data_dict.items():
                if name in names:
                    data_list.append(
                        self.parent.data[name][info[0] : info[0] + info[1]]
                    )
                    no_data_values.append(self.nan_value_from_name(name))

            data_list = self._pad_arrays_to_association(
                object_, data_list, no_data_values
            )

            if drillholes:
                # add the object list to the first position of the data list
                data_list.insert(0, [object_] * data_list[0].shape[0])

            # create a numpy array
            all_data_list.append(np.array(data_list, dtype=object).T)

        # get the names of the data
        if drillholes:
            names = ("Drillhole",) + names

        # transform to a structured array
        return self._create_structured_array(
            np.concatenate(all_data_list, axis=0), names
        )

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

    def _pad_arrays_to_association(
        self, drillhole_uid: bytes, arrays: list[np.ndarray], ndv: Any
    ) -> list[np.ndarray]:
        """
        Pad the arrays in the list to match the size of the association
            for a given drillhole.

        :param drillhole_uid: The uid of the drillhole where the data is located.
        :param arrays: The list of arrays to pad.
        :param ndv: The No Data Value of the data.

        :return: The padded arrays.
        """
        padded_arrays = []  # First array remains the same
        for idx, array in enumerate(arrays):
            pad_size = (
                self.index_by_drillhole[drillhole_uid][self.association[0]][1]
                - array.shape[0]
            )
            if pad_size > 0:
                padded_array = np.pad(
                    array,
                    (0, pad_size),
                    mode="constant",
                    constant_values=(ndv[idx], ndv[idx]),
                )
            else:
                padded_array = array
            padded_arrays.append(padded_array)

        return padded_arrays

    def add_values_to_property_group(
        self, name: str, values: np.ndarray, value_map: dict[int, str] | None = None
    ):
        """
        Push the values to each drillhole of the property group based on association.

        :param name: The name of the data to push.
        :param values: The values to push.
        :param value_map: The value map associating the index and the description
            in the case of referenced data
        """
        if not isinstance(name, str) or name in self.parent.data:
            raise KeyError("The name must be a string not present in data.")

        # ensure the length of the values is the same as the length of the template
        if values.shape != self.parent.data[self.association[0]].shape:
            raise ValueError(
                "The length of the values must be the same as the association "
                f"({self.parent.data[self.association[0]].shape})."
            )

        attributes = {}
        if isinstance(value_map, dict):
            attributes.update({"type": "referenced", "value_map": value_map})

        for drillhole_uid, indices in self.index_by_drillhole.items():
            # get the drillhole
            drillhole: ConcatenatedDrillhole = self.parent.workspace.get_entity(  # type: ignore
                str2uuid(drillhole_uid)
            )[
                0
            ]

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
                            ]
                        },
                        **attributes,
                    },
                },
                property_group=self.property_groups[drillhole.uid],
            )

        self._update_drillholes_group_table(name)

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
        self,
        names: tuple[str] | str,
        spatial_index: bool = False,
    ) -> np.ndarray:
        """
        Get a table with specific data associated with depth for every drillhole object.

        :param names: The names to extract.
        :param spatial_index: If True, the spatial index is added to the table.

        :return: a table containing the Drillholes, the association and the data.
        """
        names = to_tuple(names)

        # ensure names are in properties
        if not all(name in self.properties for name in names):
            raise KeyError("The names are not in the list of properties.")

        if spatial_index:
            return self._depth_table_by_key(self.association + names, True)

        return self._depth_table_by_key(names, False)

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
                raise AssertionError("No drillhole found in the concatenator.")

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
            properties: tuple = ()
            for property_group in self.property_groups.values():
                if property_group.properties is None:
                    continue

                temp_properties = tuple(
                    property_group.parent.get_data(property_)[0].name
                    for property_ in property_group.properties
                )

                properties = tuple(sorted(set(properties + temp_properties)))

            properties = tuple(
                name for name in properties if name not in self.association
            )

            if properties:
                self._properties = properties

        if self._properties is None:
            return ()

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

    def _update_drillholes_group_table(self, name):
        """
        Update the drillholes group table with a new property group.

        :param name: The name of the property group to update.
        """
        self.parent.update_data_index()
        self._property_groups = self._get_property_groups(self.parent, self.name)

        if self._properties is not None:
            self._properties += (name,)
        else:
            self._properties = ((name,),)

        if self._index_by_drillhole is not None:
            for value in self._index_by_drillhole.values():
                value[name] = value[self.association[0]]
