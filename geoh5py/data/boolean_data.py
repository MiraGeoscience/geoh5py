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

from .data import PrimitiveTypeEnum
from .referenced_data import ReferencedData


class BooleanData(ReferencedData):
    """
    Data class for logical (bool) values.
    """

    def format_type(self, values: np.ndarray):
        """
        Check if the type of values is valid and coerse to type bool.
        :param values: numpy array to modify.
        :return: the formatted values.
        """
        if set(values) - {0, 1} != set():
            raise ValueError(
                f"Values provided by {self.name} are not containing only 0 or 1"
            )

        return values.astype(bool)

    @classmethod
    def primitive_type(cls) -> PrimitiveTypeEnum:
        return PrimitiveTypeEnum.BOOLEAN

    @property
    def ndv(self) -> int:
        """
        No-Data-Value
        """
        return 0
