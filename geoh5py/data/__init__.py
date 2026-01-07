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

import numpy as np

from .blob_data import BlobData
from .boolean_data import BooleanData
from .colour import Colour
from .data import Data
from .data_association_enum import DataAssociationEnum
from .data_unit import DataUnit
from .datetime_data import DatetimeData
from .filename_data import FilenameData
from .float_data import FloatData
from .geometric_data import GeometricDataConstants
from .integer_data import IntegerData
from .numeric_data import NumericData
from .reference_value_map import ReferenceValueMap
from .referenced_data import ReferencedData
from .text_data import CommentsData, MultiTextData, TextData
from .unknown_data import UnknownData
from .visual_parameters import VisualParameters


class PrimitiveTypeEnum(Enum):
    """Enum for data types."""

    BLOB = BlobData
    BOOLEAN = BooleanData
    COMMENTS = CommentsData
    COLOUR = Colour
    DATETIME = DatetimeData
    FILENAME = FilenameData
    FLOAT = FloatData
    GEOMETRIC = GeometricDataConstants
    INTEGER = IntegerData
    INVALID = type(None)
    MULTI_TEXT = MultiTextData
    REFERENCED = ReferencedData
    TEXT = TextData
    UNKNOWN = UnknownData
    VISUAL_PARAMETERS = VisualParameters
    VECTOR = type(None)


class DataTypeEnum(Enum):
    INVALID = type(None)
    INTEGER = np.int32
    FLOAT = np.float32
    TEXT = str
    REFERENCED = np.uint32  # Could represent a reference type
    FILENAME = Path
    BLOB = bytes
    VECTOR = type(None)  # Assuming a vector is a list
    DATETIME = str  # Could use datetime
    GEOMETRIC = type(None)  # For custom geometric type
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
