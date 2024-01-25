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

from copy import copy
from typing import Any
from uuid import UUID

import numpy as np

from ...data import Data, DataAssociationEnum, NumericData, TextData
from .concatenator import Concatenator
from .property_group import ConcatenatedPropertyGroup


class DrillholesConcatenator(Concatenator):
    """
    Class for concatenating drillhole data.
    """

    def _no_value_property_group(
        self, name: str
    ) -> tuple[Any, ConcatenatedPropertyGroup]:
        """
        Extract the data no data values and the property group for a given data name.

        :param name: The name of the data to extract.

        :return: The no data value and the property group
        """
        # ensure data_name is in the data
        if name not in self.data:
            raise KeyError(f"Data '{name}' not found in concatenated data.")

        # get the data of the know association
        data: ConcatenatedData = self.workspace.get_entity(  # type: ignore
            UUID(self.index[name][0][-2].decode("utf-8").strip("{}"))
        )[0].get_data(UUID(self.index[name][0][-1].decode("utf-8").strip("{}")))[0]

        # ensure the association is Depth:
        if data.association != DataAssociationEnum.DEPTH:
            raise TypeError(
                f"Data '{name}' is not associated with depth ({data.association})."
            )

        # define NDV value
        ndv = np.nan
        if isinstance(data, NumericData):
            ndv = data.nan_value
        elif isinstance(data, TextData):
            ndv = ""

        return ndv, data.property_group

    @staticmethod
    def _get_depth_association(property_group: ConcatenatedPropertyGroup) -> tuple:
        """
        Based on a PropertyGroup, it gets the name of the depth (or from to)
            associated with the data.

        :param property_group: The property group to extract.

        return: the name of the values to used for association.
        """
        if property_group.property_group_type == "Interval table":
            return property_group.from_.name, property_group.to_.name
        elif property_group.property_group_type == "Depth table":
            return (property_group.depth_.name,)

        return (None,)

    def _construct_dict_data(
        self,
        data_info: dict,
    ) -> dict:
        """
        Constructs a dictionary with the objects as keys.
        The values are (start index, n count and no data value) tuple.
        The order of the dictionary depends on the first key of 'data_info'.

        :param data_info: the dictionary containing the name of the data and the no data values.

        :return: The object dictionary {object: {Column: (start_index, n_count, ndv)}}.
        """
        object_index_dictionary: dict = {}
        for object_ in self.index[list(data_info.keys())[0]]["Object ID"]:
            data_object_info = copy(data_info)

            for column, ndv in data_info.items():
                # get the line corresponding to object
                start_index, n_count, _, _ = self.index[column][
                    self.index[column]["Object ID"] == object_
                ][0]

                # get the start index and the n count
                data_object_info[column] = (start_index, n_count, ndv)

            object_index_dictionary[object_] = data_object_info

        return object_index_dictionary

    def index_by_object(
        self,
        data_name: str | list[str],
    ) -> dict:
        """
        Constructs a dictionary with the objects as keys.
        The dictionary contains the data name as keys
            and the values are (start index, n count and no data value) tuple.

        :param data_name: The name of the data to extract.

        :return: The object dictionary {object: {Column: (start_index, n_count, ndv)}}.
        """

        # get the property group and the no data values
        data_info, property_group = self._no_value_property_group(data_name)

        # define the association based on the property group
        data_info = self._get_depth_association(data_info, property_group)

        # get the index corresponding to every object and data_name
        object_index_dictionary = self._construct_dict_data(data_info)

        return object_index_dictionary

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
        dtype = [(first_name, np.array([output[0][0]]).dtype)]

        # get the first value of the dictionary
        data_dict = object_index_dictionary[list(object_index_dictionary.keys())[0]]

        # get the data names
        data_names = list(data_dict.keys())

        for idx, data_name in enumerate(data_names):
            dtype.append((data_name, np.array([output[0][idx + 1]]).dtype))

        return np.core.records.fromarrays(output.T, dtype=dtype)

    def get_depth_table(
        self,
        data_name: str | list[str],
        pad: bool = True,
        first_name: str = "Drillhole",
    ):
        # get the dictionary
        object_index_dictionary = self.index_by_object(data_name)

        all_data_list: list = []
        for object_, data_dict in object_index_dictionary.items():
            data_list = []
            no_data_values: list = []
            for name, info in data_dict.items():
                # get the data
                data_list.append(self.data[name][info[0] : info[0] + info[1]])
                no_data_values.append(info[2])
            if pad:
                data_list = self._pad_arrays_to_first(data_list, no_data_values[1:])
            else:
                data_list = self._truncate_arrays_to_smallest(data_list)

            # add the object list to the first position of the data list
            data_list.insert(0, [object_] * data_list[0].shape[0])

            # create a numpy array
            all_data_list.append(np.array(data_list, dtype=object).T)

        # concatenate all the numpy arrays using Keys a structured array
        return self._create_structured_array(
            np.concatenate(all_data_list, axis=0),
            object_index_dictionary,
            first_name=first_name,
        )
