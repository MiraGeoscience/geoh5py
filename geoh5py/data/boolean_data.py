#  Copyright (c) 2023 Mira Geoscience Ltd.
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

import numpy as np

from .data import PrimitiveTypeEnum
from .data_type import DataType
from .reference_value_map import ReferenceValueMap
from .referenced_data import ReferencedData


class BooleanData(ReferencedData):
    """
    Data class for logical (bool) values.
    """

    def __init__(self, data_type: DataType, **kwargs):
        super().__init__(data_type, **kwargs)

        self.entity_type.value_map = ReferenceValueMap({0: "False", 1: "True"})

    @classmethod
    def primitive_type(cls) -> PrimitiveTypeEnum:
        return PrimitiveTypeEnum.BOOLEAN

    @property
    def nan_value(self):
        """No-data-value"""
        return False

    @property
    def values(self) -> np.ndarray | None:
        """
        :return: values: An array of integer values
        """
        if getattr(self, "_values", None) is None:
            values = self.workspace.fetch_values(self)

            if isinstance(values, (np.ndarray, type(None))):
                self._values = self.check_vector_length(values).astype(bool)

        return self._values

    @values.setter
    def values(self, values: np.ndarray | None):
        if isinstance(values, (np.ndarray, type(None))):
            values = self.check_vector_length(values)

        else:
            raise TypeError(
                f"Input 'values' for {self} must be of type {np.ndarray} or None."
            )

        if not (np.issubdtype(values.dtype, np.integer) or values.dtype == bool):
            raise TypeError(
                f"Input 'values' dtype must be of an integer or a boolean, find {values.dtype}."
            )

        # removing no data values from the array and replace them by "False"
        values = values.astype(np.uint8)

        if set(values) - {0, 1} != set():
            raise ValueError(
                f"Values provided by {self.name} are not containing only 0 or 1"
            )

        self._values = values.astype(bool)

        self.workspace.update_attribute(self, "values")
