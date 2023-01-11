#  Copyright (c) 2023 Mira Geoscience Ltd
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

    def copy_from_extent(
        self, bounds: np.ndarray, parent=None, copy_children: bool = True
    ) -> Group | None:
        """
        Find indices of vertices within a rectangular bounds.

        :param bounds: shape(2, 2) Bounding box defined by the South-West and
            North-East coordinates. Extents can also be provided as 3D coordinates
            with shape(2, 3) defining the top and bottom limits.
        :param attributes: Dictionary of attributes to clip by extent.
        """
        new_entity = self.copy(parent=parent, copy_children=False)
        for child in self.children:
            child.copy_from_extent(
                bounds, parent=new_entity, copy_children=copy_children
            )

        if len(new_entity.children) == 0:
            new_entity.workspace.remove_entity(new_entity)
            del new_entity
            return None

        return new_entity

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
    def extent(self):
        """
        Bounding box 3D coordinates defining the limits of the entity.
        """
        return None

    @classmethod
    def find_or_create_type(cls, workspace: workspace.Workspace, **kwargs) -> GroupType:

        return GroupType.find_or_create(workspace, cls, **kwargs)
