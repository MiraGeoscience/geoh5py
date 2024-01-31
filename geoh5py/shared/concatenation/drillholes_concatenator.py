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

import numpy as np

from ...groups import DrillholeGroup
from ..utils import to_tuple
from .concatenator import Concatenator
from .drillholes_group_table import DrillholesGroupTable


class DrillholesConcatenator(Concatenator, DrillholeGroup):
    """
    Class for concatenating drillhole data.
    """

    def get_drillhole_group_table(self, data_name: tuple[str]) -> DrillholesGroupTable:
        """
        Get a DrillholesGroupTable object for the specified data,
        containing the property group information.

        :param data_name: The name of the data to extract.

        :return: The property group information.
        """
        property_groups = []
        for name in data_name:
            if name not in self.property_group_by_data_name:
                raise KeyError(f"The name '{name}' is not in the concatenator.")
            property_groups.append(self.property_group_by_data_name[name])

        # ensure there is a unique property group
        if len(set(property_groups)) != 1:
            raise ValueError("All the data must be in the same property group.")

        return DrillholesGroupTable(self, property_groups[0])

    def get_depth_table(
        self,
        data_name: tuple[str],
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
        data_name = to_tuple(data_name)
        drillholes_group_table = self.get_drillhole_group_table(data_name)
        object_index_dictionary = drillholes_group_table.index_by_drillholes()

        all_data_list = []
        for object_, data_dict in object_index_dictionary.items():
            data_list: list = []
            no_data_values: list = []

            for name, info in data_dict.items():
                if name in data_name + drillholes_group_table.association_names:
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

        # get the names of the data
        names = [first_name] + [
            name
            for name in object_index_dictionary[
                list(object_index_dictionary.keys())[0]
            ].keys()
            if name in data_name + drillholes_group_table.association_names
        ]

        # transform to a structured array
        structured_array = self._create_structured_array(
            np.concatenate(all_data_list, axis=0), names
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
    def _create_structured_array(output: np.ndarray, names: list[str]) -> np.ndarray:
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
