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

from ..utils import to_tuple
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
        self._association: tuple = ()
        self._association_type: str = ""
        self._depth_table: np.ndarray = np.array([])
        self._index_by_drillhole: dict = {}
        self._properties: tuple = ()

        self._property_groups: list = self._get_property_groups(parent, name)
        self._parent: Concatenator = parent
        self._name: str = name

        self._define_association()

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

    def _define_association(self):
        if self.property_group_type == "Interval table":
            self._association = (
                self.property_groups[0].from_.name,
                self.property_groups[0].to_.name,
            )
            self._association_type = "from-to"
        elif self.property_group_type == "Depth table":
            self._association = (self.property_groups[0].depth_.name,)
            self._association_type = "depth"
        else:
            raise TypeError(
                f"The property group type '{self.property_group_type}' is not supported."
            )

    @staticmethod
    def _get_property_groups(parent, name) -> list:
        """
        Get all the property groups with a given name in the concatenator.

        :param parent: the concatenator object.
        :param name: the name to get the property groups for.

        :return: a list of property groups.
        """
        if not hasattr(parent, "property_group_ids"):
            raise TypeError("The parent must be a Concatenator instance.")

        property_groups = []
        for property_group_uid in parent.property_group_ids:
            property_group = parent.workspace.get_entity(
                UUID(property_group_uid.decode("utf-8"))
            )[0]
            if (
                isinstance(property_group, ConcatenatedPropertyGroup)
                and property_group.name == name
            ):
                property_groups.append(property_group)

        if not property_groups:
            raise ValueError(
                f"No property group with name '{name}' found in the concatenator."
            )

        return property_groups

    @staticmethod
    def _pad_arrays_to_first(arrays: list[np.ndarray], ndv: Any) -> list[np.ndarray]:
        """
        Pad the arrays in the list to match the size of the first array.

        :param arrays: The list of arrays to pad.
        :param ndv: The No Data Value of the data.

        :return: The padded arrays.
        """
        # Pad other arrays in the list to match the size of the first array
        padded_arrays = [arrays[0]]  # First array remains the same
        for idx, array in enumerate(arrays[1:]):
            pad_size = arrays[0].shape[0] - array.shape[0]
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
        self,
        name: str,
        values: np.ndarray,
    ):
        """
        Push the values to each drillhole of the property group based on association.

        :param name: The name of the data to push.
        :param values: The values to push.
        """
        if not isinstance(name, str) or name in self.parent.data:
            raise KeyError("The name must be a string not present in data.")

        # ensure the length of the values is the same as the length of the template
        if values.shape != self.parent.data[self.association[0]].shape:
            raise ValueError(
                "The length of the values must be the same as the association "
                f"({self.parent.data[self.association[0]].shape})."
            )

        for drillhole_uid, indices in self.index_by_drillhole.items():
            # get the drillhole
            drillhole: ConcatenatedDrillhole = self.parent.workspace.get_entity(  # type: ignore
                UUID(drillhole_uid.decode("utf-8"))
            )[
                0
            ]

            # define the associations values
            drillhole_association = []
            for key in self.association:
                drillhole_association.append(
                    self.parent.data[key][
                        indices[key][0] : indices[key][0] + indices[key][1]
                    ]
                )

            # add data to the drillhole
            drillhole.add_data(
                {
                    name: {
                        "values": values[
                            indices[self.association[0]][0] : indices[
                                self.association[0]
                            ][0]
                            + indices[self.association[0]][1]
                        ],
                        self.association_type: np.array(drillhole_association).T,
                    },
                },
                property_group=self.name,
            )

    @property
    def association(self) -> tuple:
        """
        The depth association of the PropertyGroup.
        """
        return self._association

    @property
    def association_type(self) -> str:
        """
        The type of the association.
        """
        return self._association_type

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
        if not self._depth_table.size:
            all_data_list = []
            for object_, data_dict in self.index_by_drillhole.items():
                data_list: list = []
                no_data_values: list = []

                for name, info in data_dict.items():
                    data_list.append(
                        self.parent.data[name][info[0] : info[0] + info[1]]
                    )
                    no_data_values.append(self.nan_value_from_name(name))

                data_list = self._pad_arrays_to_first(data_list, no_data_values[1:])

                # add the object list to the first position of the data list
                data_list.insert(0, [object_] * data_list[0].shape[0])

                # create a numpy array
                all_data_list.append(np.array(data_list, dtype=object).T)

            # get the names of the data
            names = ("Drillhole",) + self.association + self.properties

            # transform to a structured array
            self._depth_table = self._create_structured_array(
                np.concatenate(all_data_list, axis=0), names
            )

        return self._depth_table

    def depth_table_by_name(
        self,
        names: tuple[str],
    ) -> np.ndarray:
        """
        Get a table with specific data associated with depth for every drillhole object.

        :param names: The names to extract.

        :return: a table containing the Drillholes, the association and the data.
        """
        names = to_tuple(names)

        # ensure names are in properties
        if not all(name in self.properties for name in names):
            raise KeyError("The names are not in the list of properties.")

        return self.depth_table[[*("Drillhole",) + self.association + names]]

    @property
    def index_by_drillhole(
        self,
    ) -> dict:
        """
        Get for every object index and count of all the data in 'association' and 'properties'

        :return: A dictionary with the object uuid and the index of all the data.
        """
        if not self._index_by_drillhole:
            names = self.association + self.properties
            for drillhole in np.sort(self._parent.index[names[0]], order="Start index")[
                "Object ID"
            ]:
                self._index_by_drillhole[drillhole] = {}
                for name in names:
                    if drillhole in self._parent.index[name]["Object ID"]:
                        self._index_by_drillhole[drillhole][name] = list(
                            self._parent.index[name][
                                self._parent.index[name]["Object ID"] == drillhole
                            ][0]
                        )[:2]
                    else:
                        self._index_by_drillhole[drillhole][name] = [0, 0]

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
            UUID(self.parent.index[name][0][-2].decode("utf-8").strip("{}"))
        )[0].get_data(UUID(self.parent.index[name][0][-1].decode("utf-8").strip("{}")))[
            0
        ]

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
            for property_group in self.property_groups:
                if property_group.properties is None:
                    continue

                temp_properties = tuple(
                    property_group.parent.get_data(property_)[0].name
                    for property_ in property_group.properties
                )

                self._properties = tuple(
                    sorted(set(self._properties + temp_properties))
                )

            self._properties = tuple(
                name for name in self._properties if name not in self.association
            )

        return self._properties

    @property
    def property_group_type(self) -> str:
        """
        The type of the PropertyGroup.
        """
        return self.property_groups[0].property_group_type

    @property
    def property_groups(self) -> list:
        """
        Get all the property groups in the concatenator.

        :return: A list containing all the property groups.
        """
        return self._property_groups