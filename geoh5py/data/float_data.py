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

from ..shared import FLOAT_NDV
from .data import DataType, PrimitiveTypeEnum
from .numeric_data import NumericData


class FloatData(NumericData):
    """
    Data container for floats values
    """

    def __init__(self, data_type: DataType, **kwargs):
        super().__init__(data_type, **kwargs)

    @classmethod
    def primitive_type(cls) -> PrimitiveTypeEnum:
        return PrimitiveTypeEnum.FLOAT

    @classmethod
    def ndv(cls) -> float:
        """
        No-Data-Value
        """
        return FLOAT_NDV
