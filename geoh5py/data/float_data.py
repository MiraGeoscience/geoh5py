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

import numpy as np

from ..shared import FLOAT_NDV
from .data import PrimitiveTypeEnum
from .numeric_data import NumericData


class FloatData(NumericData):
    """
    Data container for floats values
    """

    def format_type(self, values: np.ndarray) -> np.ndarray:
        """
        Check if the type of values is valid and coerse to type float64.
        :param values: numpy array to modify.
        :return: the formatted values.
        """
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError("Values must be a numpy array of numeric values.")

        return values

    @property
    def formatted_values(self):
        values = self.ndv_values
        if values is None:
            return values

        return values

    @classmethod
    def primitive_type(cls) -> PrimitiveTypeEnum:
        return PrimitiveTypeEnum.FLOAT

    @property
    def nan_value(self):
        """
        Nan-Data-Value
        """
        return np.nan

    @property
    def ndv(self) -> float:
        """
        No-Data-Value
        """
        return FLOAT_NDV
