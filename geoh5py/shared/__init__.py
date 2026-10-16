# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2020-2026 Mira Geoscience Ltd.                                '
#                                                                              '
#  This file is part of geoh5py.                                               '
#                                                                              '
#  geoh5py is free software: you can redistribute it and/or modify             '
#  it under the terms of the GNU Lesser General Public License as published by '
#  the Free Software Foundation, either version 3 of the License, or           '
#  (at your option) any later version.                                         '
#                                                                              '
#  geoh5py is distributed in the hope that it will be useful,                  '
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              '
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               '
#  GNU Lesser General Public License for more details.                         '
#                                                                              '
#  You should have received a copy of the GNU Lesser General Public License    '
#  along with geoh5py.  If not, see <https://www.gnu.org/licenses/>.           '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from geoh5py.data import PrimitiveTypeEnum

INTEGER_NDV = -2147483648
FLOAT_NDV = 1.17549435e-38


class DataTypeEnum(Enum):
    INVALID = type(None)  # type: ignore[misc]
    INTEGER = np.int32
    FLOAT = np.float32
    TEXT = str
    REFERENCED = np.uint32  # Could represent a reference type
    FILENAME = Path
    BLOB = bytes
    VECTOR = type(None)  # type: ignore[misc]
    DATETIME = str  # Could use datetime
    GEOMETRIC = type(None)  # type: ignore[misc]
    MULTI_TEXT = str
    BOOLEAN = bool

    @classmethod
    def from_primitive_type(cls, primitive_type: PrimitiveTypeEnum) -> type:
        """
        Get the data type from the primitive type.

        :param primitive_type: The primitive type.
        :return: The data type.
        """
        return DataTypeEnum[primitive_type.name].value
