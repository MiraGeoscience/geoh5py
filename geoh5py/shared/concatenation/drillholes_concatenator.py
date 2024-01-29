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

from typing import Any
from uuid import UUID

import numpy as np

from ...groups import DrillholeGroup
from .concatenator import Concatenator
from .data import ConcatenatedData
from .property_group import ConcatenatedPropertyGroup


class DrillholesConcatenator(Concatenator, DrillholeGroup):
    """
    Class for concatenating drillhole data.
    """

    def get_data_from_name(self, name: str) -> ConcatenatedData:
        """
        Get the data from a given name.

        :param name: The name of the data to extract.

        :return: The first data found with the given name in index.
        """

        # ensure data_name is in the data
        if name not in self.data:
            raise KeyError(f"Data '{name}' not found in concatenated data.")

        # get the data of the know association
        data: ConcatenatedData = self.workspace.get_entity(  # type: ignore
            UUID(self.index[name][0][-2].decode("utf-8").strip("{}"))
        )[0].get_data(UUID(self.index[name][0][-1].decode("utf-8").strip("{}")))[0]

        return data

    @staticmethod
    def get_depth_association(
        property_group: ConcatenatedPropertyGroup,
    ) -> tuple[str] | tuple[str, str] | None:
        """
        Based on a PropertyGroup, it gets the name of the depth (or from to)
            associated with the data.

        :param property_group: The property group to extract.

        return: the name of the values to used for association.
        """
        if getattr(property_group, "property_group_type", None) == "Interval table":
            return property_group.from_.name, property_group.to_.name
        if getattr(property_group, "property_group_type", None) == "Depth table":
            return (property_group.depth_.name,)

        return None

    def association_by_drillhole(
        self,
        names: tuple,
    ) -> dict:
        """
        Based on the first name of the input list,
        get for every object index and count of all the data in 'names'

        :param names: The names of the data to extract.

        :return: A dictionary with the object uuid and the index of all the data.
        """
        # ensure all the names are in the index
        if not all(name in self.index for name in names):
            raise KeyError(f"Data '{names}' not found in concatenated data.")

        object_index: dict = {}
        for drillhole in self.index[names[0]]["Object ID"]:
            object_index[drillhole] = {}
            for name in names:
                if drillhole in self.index[name]["Object ID"]:
                    object_index[drillhole][name] = list(
                        self.index[name][self.index[name]["Object ID"] == drillhole][0]
                    )[:2]
                else:
                    object_index[drillhole][name] = [0, 0]

        return object_index

    def depth_single_association(self, name: str) -> dict:
        """
        Get the index and N count of the data associated with depth for every drillhole object.

        :param name: The name of the data to extract.

        :return: A dictionary with the object uuid, the association and the data index.
        """

        data = self.get_data_from_name(name)

        association_name = None
        if data.property_group is not None:
            association_name = self.get_depth_association(data.property_group)
        if association_name is None:
            raise ValueError(f"Data '{name}' is not associated with depth.")

        # merge association with name at the end
        associations = self.association_by_drillhole(association_name + (name,))

        return associations

    def depth_multiple_association(self, names: str | tuple[str] | list[str]) -> dict:
        """
        Get the index and N count of the data associated with depth for every drillhole object.
        The data must have the same association. It runs the function depth_single_association
        for every data in the list and ensure the association is the same.

        :param names: The names of the data to extract.

        :return: A dictionary with the object uuid, the association and the data index.
        """
        if not isinstance(names, (list, tuple)):
            names = [names]

        associations = self.depth_single_association(names[0])

        for name in names[1:]:
            association = self.depth_single_association(name)

            # ensure the first value is the same
            if (
                list(list(association.values())[0].items())[0]
                != list(list(associations.values())[0].items())[0]
            ):
                raise AssertionError(f"Data '{names}' don't have the same association.")

            # update the dictionary
            for drillhole, association_dict in associations.items():
                association_dict.update(association[drillhole])

        return associations

    def get_depth_table(
        self,
        data_name: str | list[str],
        pad: bool = True,
        first_name: str = "Drillhole",
    ):
        """
        Get a table with all the data associated with depth for every drillhole object.
        The Drillhole name is added at the beginning of the table for every row.
        Several data can be extracted at the same time and push in the same table.

        :param data_name: The name of the data to extract.
        :param pad: If True, the data are padded to the size of the first data;
            otherwise, the data are truncated to the size of the smallest data.
        :param first_name: The name of the first column of the structured array.

        :return: a structured array with all the data.
        """
        # get the dictionary
        object_index_dictionary = self.depth_multiple_association(data_name)

        all_data_list = []
        for object_, data_dict in object_index_dictionary.items():
            data_list: list = []
            no_data_values: list = []
            # get the values
            for name, info in data_dict.items():
                data_list.append(self.data[name][info[0] : info[0] + info[1]])
                if pad:
                    no_data_values.append(self.get_data_from_name(name).nan_value)

            # transform all the data to the same size
            if pad:
                data_list = self._pad_arrays_to_first(data_list, no_data_values[1:])
            else:
                data_list = self._truncate_arrays_to_smallest(data_list)

            # add the object list to the first position of the data list
            data_list.insert(0, [object_] * data_list[0].shape[0])

            # create a numpy array
            all_data_list.append(np.array(data_list, dtype=object).T)

        # concatenate all the numpy arrays using Keys a structured array
        structured_array = self._create_structured_array(
            np.concatenate(all_data_list, axis=0),
            object_index_dictionary,
            first_name=first_name,
        )

        return structured_array

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

    @staticmethod
    def _truncate_arrays_to_smallest(arrays: list[np.ndarray]) -> list[np.ndarray]:
        """
        Truncate the arrays in the list to match the size of the smallest array.

        :param arrays: The list of arrays to truncate.

        :return: The truncated arrays.
        """
        # Find the size of the smallest array
        smallest_size = min(array.shape[0] for array in arrays)

        # Truncate each array to the size of the smallest one
        truncated_arrays = [array[:smallest_size] for array in arrays]

        return truncated_arrays

    @staticmethod
    def _create_structured_array(
        output: np.ndarray, object_index_dictionary: dict, first_name: str = "Drillhole"
    ) -> np.ndarray:
        """
        Create a structured array from the output of the function get_depth_table.

        :param output: The data to pass to structured array.
        :param object_index_dictionary: The dictionary to get the data names.
        :param first_name: The name of the first column of the structured array.

        :return: The structured array.
        """
        # create the structured array
        dtype = [(first_name, "O")]

        for idx, data_name in enumerate(
            list(
                object_index_dictionary[list(object_index_dictionary.keys())[0]].keys()
            )
        ):
            type_temp = np.array([output[0, idx + 1]]).dtype
            if type_temp.kind in ["S", "U"]:
                dtype.append((data_name, "O"))
            else:
                dtype.append((data_name, type_temp))

        return np.core.records.fromarrays(output.T, dtype=dtype)
