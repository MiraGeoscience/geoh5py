# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2025-2026 Mira Geoscience Ltd.                                '
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
from typing import TYPE_CHECKING

from .object_base import ObjectBase


if TYPE_CHECKING:
    from numpy import ndarray


class Label(ObjectBase):
    """
    Label object for annotation in viewport.

    .. warning:: Not yet implemented.

    """

    _TYPE_UID = uuid.UUID(
        fields=(0xE79F449D, 0x74E3, 0x4598, 0x9C, 0x9C, 0x351A28B8B69E)
    )

    def __init__(self, **kwargs):
        # TODO
        self.target_position = None
        self.label_position = None

        super().__init__(**kwargs)

    @property
    def extent(self):
        """
        Geography bounding box of the object.
        """
        return None

    def mask_by_extent(self, extent: ndarray, inverse: bool = False) -> None:
        """
        Sub-class extension of :func:`~geoh5py.shared.entity.Entity.mask_by_extent`.
        """
        return None
