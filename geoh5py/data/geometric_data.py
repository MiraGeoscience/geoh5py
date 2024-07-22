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

import uuid

from .data import Data
from .data_type import DataType
from .primitive_type_enum import PrimitiveTypeEnum


class GeometricDataConstants(Data):
    """
    Base class for geometric data constants.
    """

    _TYPE_UID: uuid.UUID

    def __init__(self, **kwargs):
        super().__init__(allow_move=False, visible=False, **kwargs)

    @classmethod
    def primitive_type(cls) -> PrimitiveTypeEnum:
        return PrimitiveTypeEnum.GEOMETRIC


class GeometricDataConstantsX(DataType):
    """
    Data container for X values
    """

    _TYPE_UID = uuid.UUID(
        fields=(0xE9E6B408, 0x4109, 0x4E42, 0xB6, 0xA8, 0x685C37A802EE)
    )

    @classmethod
    def default_type_uid(cls) -> uuid.UUID | None:
        """
        Default uuid for the entity type.
        """
        return cls._TYPE_UID


class GeometricDataConstantsY(DataType):
    """
    Data container for Y values
    """

    _TYPE_UID = uuid.UUID(
        fields=(0xF55B07BD, 0xD8A0, 0x4DFF, 0xBA, 0xE5, 0xC975D490D71C)
    )

    @classmethod
    def default_type_uid(cls) -> uuid.UUID | None:
        """
        Default uuid for the entity type.
        """
        return cls._TYPE_UID


class GeometricDataConstantsZ(DataType):
    """
    Data container for X values
    """

    _TYPE_UID = uuid.UUID(
        fields=(0xDBAFB885, 0x1531, 0x410C, 0xB1, 0x8E, 0x6AC9A40B4466)
    )

    @classmethod
    def default_type_uid(cls) -> uuid.UUID | None:
        """
        Default uuid for the entity type.
        """
        return cls._TYPE_UID
