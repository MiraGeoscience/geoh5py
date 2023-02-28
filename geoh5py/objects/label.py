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

import uuid
from typing import TYPE_CHECKING

from .object_base import ObjectBase, ObjectType

if TYPE_CHECKING:
    from numpy import ndarray


class Label(ObjectBase):
    """
    Label object for annotation in viewport.

    .. warning:: Not yet implemented.

    """

    __TYPE_UID = uuid.UUID(
        fields=(0xE79F449D, 0x74E3, 0x4598, 0x9C, 0x9C, 0x351A28B8B69E)
    )

    def __init__(self, object_type: ObjectType, **kwargs):
        # TODO
        self.target_position = None
        self.label_position = None

        super().__init__(object_type, **kwargs)

        object_type.workspace._register_object(self)

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID

    @property
    def extent(self):
        """
        Geography bounding box of the object.
        """
        return None

    def mask_by_extent(
        self,
        extent: ndarray,
    ) -> ndarray | None:
        """
        Find indices of vertices or centroids within a rectangular extent.

        :param extent: shape(2, 2) Bounding box defined by the South-West and
            North-East coordinates. Extents can also be provided as 3D coordinates
            with shape(2, 3) defining the top and bottom limits.
        """
        return None
