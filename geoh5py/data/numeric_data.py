#  Copyright (c) 2022 Mira Geoscience Ltd.
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
from abc import ABC

import numpy as np

from .data import Data, DataType, PrimitiveTypeEnum


class NumericData(Data, ABC):
    """
    Data container for floats values
    """

    def __init__(self, data_type: DataType, **kwargs):
        super().__init__(data_type, **kwargs)

    @classmethod
    def primitive_type(cls) -> PrimitiveTypeEnum:
        return PrimitiveTypeEnum.INVALID

    @property
    def values(self) -> np.ndarray:
        """
        :return: values: An array of float values
        """
        if getattr(self, "_values", None) is None:
            if self.concatenated:
                self._values = self.parent.get_concatenated_data(self)
            elif self.on_file:
                self._values = self.workspace.fetch_values(self.uid)

        if self._values is not None:
            self._values = self.check_vector_length(self._values)

        return self._values

    @values.setter
    def values(self, values):
        self._values = self.check_vector_length(values)
        self.workspace.update_attribute(self, "values")

    def check_vector_length(self, values) -> np.ndarray:
        """
        Check for possible mismatch between the length of values
        stored and the expected number of cells or vertices.
        """
        if self.n_values is not None and (
            values is None or len(values) < self.n_values
        ):
            full_vector = np.ones(self.n_values) * np.nan
            full_vector[: len(np.ravel(values))] = np.ravel(values)

            return full_vector

        return values

    def __call__(self):
        return self.values
