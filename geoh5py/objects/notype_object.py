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
import warnings
from typing import TYPE_CHECKING

from .object_base import ObjectBase
from .object_type import ObjectType

if TYPE_CHECKING:
    from numpy import ndarray


class NoTypeObject(ObjectBase):
    """
    Generic Data object without a registered type
    """

    __TYPE_UID = uuid.UUID(
        fields=(0x849D2F3E, 0xA46E, 0x11E3, 0xB4, 0x01, 0x2776BDF4F982)
    )

    def __init__(self, object_type: ObjectType, **kwargs):
        super().__init__(object_type, **kwargs)

        object_type.workspace._register_object(self)

    def copy(
        self,
        parent=None,
        copy_children: bool = True,
        clear_cache: bool = False,
        mask: ndarray | None = None,
        cell_mask: ndarray | None = None,
        **kwargs,
    ):
        """
        Function to copy an entity to a different parent entity.

        :param parent: Target parent to copy the entity under. Copied to current
            :obj:`~geoh5py.shared.entity.Entity.parent` if None.
        :param copy_children: (Optional) Create copies of all children entities along with it.
        :param clear_cache: Clear array attributes after copy.
        :param mask: Array of indices to sub-sample the input entity.
        :param cell_mask: Array of indices to sub-sample the input entity cells.
        :param kwargs: Additional keyword arguments.

        :return: New copy of the input entity.
        """
        if mask is not None or cell_mask is not None:
            warnings.warn("Masking is not supported for NoType objects.")

        new_entity = super().copy(
            parent=parent,
            copy_children=copy_children,
            clear_cache=clear_cache,
            **kwargs,
        )

        return new_entity

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
    ) -> None:
        """
        Sub-class extension of :func:`~geoh5py.shared.entity.Entity.mask_by_extent`.
        """
        return None
