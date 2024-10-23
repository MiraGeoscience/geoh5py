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

from typing import TYPE_CHECKING

import numpy as np

from ..shared.entity_container import EntityContainer
from .group_type import GroupType


if TYPE_CHECKING:
    from ..workspace import Workspace


class Group(EntityContainer):
    """Base Group class"""

    _default_name = "Group"

    def mask_by_extent(self, extent: np.ndarray, inverse: bool = False) -> None:
        """
        Sub-class extension of :func:`~geoh5py.shared.entity.Entity.mask_by_extent`.
        """
        return None

    def copy(
        self,
        parent=None,
        *,
        copy_children: bool = True,
        clear_cache: bool = False,
        mask: np.ndarray | None = None,
        **kwargs,
    ):
        """
        Function to copy a group to a different parent entity.

        :param parent: Target parent to copy the entity under. Copied to current
            :obj:`~geoh5py.shared.entity.Entity.parent` if None.
        :param copy_children: (Optional) Create copies of all children entities along with it.
        :param clear_cache: Clear array attributes after copy.
        :param mask: Array of bool defining the values to keep.
        :param kwargs: Additional keyword arguments to pass to the copy constructor.

        :return entity: Registered Entity to the workspace.
        """
        if parent is None:
            parent = self.parent

        new_entity = parent.workspace.copy_to_parent(
            self, parent, copy_children=False, **kwargs
        )

        if new_entity is None:
            return None

        if copy_children:
            for child in self.children:
                child.copy(
                    parent=new_entity,
                    copy_children=True,
                    clear_cache=clear_cache,
                    mask=mask,
                )

        return new_entity

    def copy_from_extent(
        self,
        extent: np.ndarray,
        parent=None,
        *,
        copy_children: bool = True,
        clear_cache: bool = False,
        inverse: bool = False,
        **kwargs,
    ) -> Group | None:
        """
        Sub-class extension of :func:`~geoh5py.shared.entity.Entity.copy_from_extent`.
        """
        copy_group = self.copy(
            parent=parent,
            clear_cache=clear_cache,
            copy_children=False,
            **kwargs,
        )

        if copy_group is None:
            return None

        if copy_children:
            for child in self.children:
                child.copy_from_extent(
                    extent,
                    parent=copy_group,
                    copy_children=True,
                    clear_cache=clear_cache,
                    inverse=inverse,
                )

            if len(copy_group.children) == 0:
                copy_group.workspace.remove_entity(copy_group)
                return None

        return copy_group

    @property
    def entity_type(self) -> GroupType:
        return self._entity_type

    @property
    def extent(self) -> np.ndarray | None:
        """
        Geography bounding box of the object.

        :return: shape(2, 3) Bounding box defined by the bottom South-West and
            top North-East coordinates.
        """
        extents = []
        for child in self.children:
            if child.extent is not None:
                extents.append(child.extent)

        if len(extents) > 0:
            extents = np.vstack(extents)
            return np.vstack(
                [
                    np.min(extents, axis=0),
                    np.max(extents, axis=0),
                ]
            )

        return None

    @classmethod
    def find_or_create_type(cls, workspace: Workspace, **kwargs) -> GroupType:
        kwargs["entity_class"] = cls
        return GroupType.find_or_create(workspace, **kwargs)

    def validate_entity_type(self, entity_type: GroupType | None) -> GroupType:
        """
        Validate the entity type.
        """
        if not isinstance(entity_type, GroupType):
            raise TypeError(
                f"Input 'entity_type' must be of type {GroupType}, not {type(entity_type)}"
            )

        if entity_type.name == "Entity":
            entity_type.name = self.name

        return entity_type
