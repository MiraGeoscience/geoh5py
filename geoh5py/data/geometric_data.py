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
from .primitive_type_enum import PrimitiveTypeEnum


class GeometricDataConstants(Data):
    """
    Base class for geometric data constants.

    :param allow_move: Defaults coordinate to remain on object.
    :param visible: Defaults to not visible.
    """

    _TYPE_UID: uuid.UUID

    def __init__(self, allow_move=False, visible=False, **kwargs):
        super().__init__(allow_move=allow_move, visible=visible, **kwargs)

    @classmethod
    def primitive_type(cls) -> PrimitiveTypeEnum:
        return PrimitiveTypeEnum.GEOMETRIC