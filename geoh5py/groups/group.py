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
from abc import abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from ..data import CommentsData, Data
from ..shared import Entity
from .group_type import GroupType

if TYPE_CHECKING:
    from .. import workspace


class Group(Entity):
    """Base Group class"""

    def __init__(self, group_type: GroupType, **kwargs):
        assert group_type is not None
        self._entity_type = group_type
        super().__init__(**kwargs)

    @classmethod
    @abstractmethod
    def default_type_uid(cls) -> uuid.UUID | None:
        ...

    def add_comment(self, comment: str, author: str | None = None):
        """
        Add text comment to an object.

        :param comment: Text to be added as comment.
        :param author: Author's name or :obj:`~geoh5py.workspace.workspace.Worspace.contributors`.
        """

        date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        if author is None:
            author = ",".join(self.workspace.contributors)

        comment_dict = {"Author": author, "Date": date, "Text": comment}

        if self.comments is None:
            self.workspace.create_entity(
                Data,
                entity={
                    "name": "UserComments",
                    "values": [comment_dict],
                    "parent": self,
                    "association": "OBJECT",
                },
                entity_type={"primitive_type": "TEXT"},
            )

        else:
            self.comments.values = self.comments.values + [comment_dict]

    def mask_by_extent(self, extent: np.ndarray, inverse: bool = False) -> None:
        """
        Sub-class extension of :func:`~geoh5py.shared.entity.Entity.mask_by_extent`.
        """

        return None

    def copy(
        self,
        parent=None,
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
    def comments(self):
        """
        Fetch a :obj:`~geoh5py.data.text_data.CommentsData` entity from children.
        """
        for child in self.children:
            if isinstance(child, CommentsData):
                return child

        return None

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
    def find_or_create_type(cls, workspace: workspace.Workspace, **kwargs) -> GroupType:
        return GroupType.find_or_create(workspace, cls, **kwargs)
