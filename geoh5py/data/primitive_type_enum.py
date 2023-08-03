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

from enum import Enum

import numpy as np

from ..shared import INTEGER_NDV


class PrimitiveTypeEnum(Enum):
    """
    Known data type.

    Available options:
    """

    INVALID = 0
    INTEGER = 1
    FLOAT = 2
    TEXT = 3
    REFERENCED = 4
    FILENAME = 5
    BLOB = 6
    VECTOR = 7
    DATETIME = 8
    GEOMETRIC = 9
    MULTI_TEXT = 10
    BOOLEAN = 11


def convert_to_primitive_type(data: np.ndarray, primitive_type: str) -> np.ndarray:
    """
    Convert a numpy array to a primitive type.

    :param data: numpy array to convert
    :param primitive_type: type to convert to

    :return: numpy array of primitive type
    """
    if isinstance(data, np.ndarray):
        if primitive_type in ["INTEGER", "REFERENCED"]:
            data[np.isnan(data)] = INTEGER_NDV
            return data.astype(np.int32)
        if primitive_type == "BOOLEAN":
            data[np.isnan(data)] = 0
            return data.astype(bool)

    return data
