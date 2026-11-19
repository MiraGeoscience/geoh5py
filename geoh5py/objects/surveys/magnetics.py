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


class AirborneMagnetics(Curve):
    """
    An airborne magnetic survey object.

    .. warning:: Partially implemented.

    """

    _TYPE_UID = uuid.UUID("4b99204c-d133-4579-a916-a9c8b98cfccb")
    _default_name = "Survey Airborne Magnetics"


class GroundMagnetics(Curve):
    """
    A ground magnetic survey object.

    .. warning:: Partially implemented.

    """

    _TYPE_UID = uuid.UUID("{028e4905-cc97-4dab-b1bf-d76f58b501b5}")
    _default_name = "Survey Ground Magnetics"
