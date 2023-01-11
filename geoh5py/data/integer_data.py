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

import warnings

import numpy as np

from ..shared import INTEGER_NDV
from .data import PrimitiveTypeEnum
from .numeric_data import NumericData


class IntegerData(NumericData):
    @classmethod
    def primitive_type(cls) -> PrimitiveTypeEnum:
        return PrimitiveTypeEnum.INTEGER

    @property
    def ndv(self) -> int:
        """
        No-Data-Value
        """
        return INTEGER_NDV

    @property
    def values(self) -> np.ndarray | None:
        """
        :return: values: An array of integer values
        """
        if getattr(self, "_values", None) is None:
            values = self.workspace.fetch_values(self)

            if isinstance(values, (np.ndarray, type(None))):
                self._values = self.check_vector_length(values)

        return self._values

    @values.setter
    def values(self, values: np.ndarray | None):
        if isinstance(values, (np.ndarray, type(None))):
            values = self.check_vector_length(values)

        else:
            raise ValueError(
                f"Input 'values' for {self} must be of type {np.ndarray} or None."
            )

        if isinstance(values, np.ndarray) and values.dtype not in [np.uint32, np.int32]:
            warnings.warn(
                f"Values provided in {values.dtype} are converted to int32 for "
                f"{self.primitive_type()} data '{self.name}.'"
            )
            values[np.isnan(values)] = self.ndv
            values = values.astype(np.int32)

        self._values = values

        self.workspace.update_attribute(self, "values")
