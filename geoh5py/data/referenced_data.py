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

from .integer_data import IntegerData
from .primitive_type_enum import PrimitiveTypeEnum


class ReferencedData(IntegerData):
    """
    Reference data described by indices and associated strings.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if "value_map" in kwargs:
            self.entity_type.value_map = kwargs["value_map"]

    @classmethod
    def primitive_type(cls) -> PrimitiveTypeEnum:
        return PrimitiveTypeEnum.REFERENCED

    @property
    def value_map(self):
        """
        Pointer to the :obj:`data.data_type.DataType.value_map`
        """
        return self.entity_type.value_map
