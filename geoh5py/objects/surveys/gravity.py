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

import uuid

from ..curve import Curve


class AirborneGravity(Curve):
    """
    An airborne gravity survey object.

    .. warning:: Partially implemented.

    """

    _TYPE_UID = uuid.UUID("{b54f6be6-0eb5-4a4e-887a-ba9d276f9a83}")
    _default_name = "Survey airborne gravity"


class GroundGravity(Curve):
    """
    A ground gravity survey object.

    .. warning:: Partially implemented.

    """

    _TYPE_UID = uuid.UUID("{5ffa3816-358d-4cdd-9b7d-e1f7f5543e05}")
    _default_name = "Survey ground gravity"
