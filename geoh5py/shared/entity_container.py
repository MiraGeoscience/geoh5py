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

# pylint: disable=R0904

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..data import CommentsData, Data
from .entity import Entity


if TYPE_CHECKING:
    from .. import shared
    from ..groups import PropertyGroup

DEFAULT_CRS = {"Code": "Unknown", "Name": "Unknown"}


class EntityContainer(Entity):
    """
    Base Entity class
    """

    _TYPE_UID: uuid.UUID | None = None

    def __init__(self, **kwargs):
        self._children: list = []
        super().__init__(**kwargs)

    @classmethod
    def default_type_uid(cls) -> uuid.UUID | None:
        """
        Default uuid for the entity type.
        """
        return cls._TYPE_UID

    def add_children(
        self, children: Entity | PropertyGroup | list[Entity | PropertyGroup]
    ):
        """
        :param children: Add a list of entities as
            :obj:`~geoh5py.shared.entity.Entity.children`
        """
        if not isinstance(children, list):
            children = [children]

        for child in children:
            if child in self._children:
                continue

            if not isinstance(child, Entity):
                raise TypeError(
                    f"Child must be an instance of Entity, not {type(child)}"
                )

            self._children.append(child)

            if hasattr(child, "parent") and child.parent is not self:
                child.parent = self

    def add_comment(self, comment: str, author: str | None = None):
        """
        Add text comment to an object.

        :param comment: Text to be added as comment.
        :param author: Author's name or :obj:`~geoh5py.workspace.workspace.Workspace.contributors`.
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
                    "association": "OBJECT",
                    "values": {"Comments": [comment_dict]},
                    "parent": self,
                },
                entity_type={"primitive_type": "TEXT"},
            )

        else:
            self.comments.values = {
                "Comments": self.comments.values["Comments"] + [comment_dict]
            }

    def add_file(self, file: str | Path | bytes, name: str = "filename.dat"):
        """
        Add a file to the object or group stored as bytes on a FilenameData

        :param file: File name with path to import.
        :param name: Name of the file in the workspace.
        """
        if isinstance(file, str):
            file = Path(file)

        if isinstance(file, Path):
            if not file.is_file():
                raise ValueError(f"Input file '{file}' does not exist.")

            name = Path(file).name

            with open(file, "rb") as raw_binary:
                blob = raw_binary.read()

        elif isinstance(file, bytes):
            blob = file

        else:
            raise TypeError(
                f"Input file must be a path or BytesIO object, not {type(file)}"
            )

        attributes = {
            "name": name,
            "file_bytes": blob,
            "association": "OBJECT",
            "parent": self,
            "values": name,
        }
        entity_type = {"name": "UserFiles", "primitive_type": "FILENAME"}

        file_data = self.workspace.create_entity(
            None, entity=attributes, entity_type=entity_type
        )

        return file_data

    @property
    def children(self):
        """
        :obj:`list` Children entities in the workspace tree
        """
        return self._children

    @property
    def comments(self):
        """
        Fetch a :obj:`~geoh5py.data.text_data.CommentsData` entity from children.
        """
        for child in self.children:
            if isinstance(child, CommentsData):
                return child

        return None

    @abstractmethod
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
        Function to copy an entity to a different parent entity.

        :param parent: Target parent to copy the entity under. Copied to current
            :obj:`~geoh5py.shared.entity.Entity.parent` if None.
        :param copy_children: (Optional) Create copies of all children entities along with it.
        :param clear_cache: Clear array attributes after copy to minimize the
            memory footprint of the workspace.
        :param mask: Array of indices to sub-sample the input entity.
        :param kwargs: Additional keyword arguments to pass to the copy constructor.

        :return entity: Registered Entity to the workspace.
        """

    def copy_from_extent(
        self,
        extent: np.ndarray,
        parent=None,
        *,
        copy_children: bool = True,
        clear_cache: bool = False,
        inverse: bool = False,
        **kwargs,
    ) -> Entity | None:
        """
        Function to copy an entity to a different parent entity.

        :param extent: Bounding box extent requested for the input entity, as supplied for
            :func:`~geoh5py.shared.entity.Entity.mask_by_extent`.
        :param parent: Target parent to copy the entity under. Copied to current
            :obj:`~geoh5py.shared.entity.Entity.parent` if None.
        :param copy_children: (Optional) Create copies of all children entities along with it.
        :param clear_cache: Clear array attributes after copy.
        :param inverse: Keep the inverse (clip) of the extent selection.
        :param kwargs: Additional keyword arguments to pass to the copy constructor.

        :return entity: Registered Entity to the workspace.
        """
        indices = self.mask_by_extent(extent, inverse=inverse)
        if indices is None:
            return None

        return self.copy(
            parent=parent,
            copy_children=copy_children,
            clear_cache=clear_cache,
            mask=indices,
            **kwargs,
        )

    def get_entity(self, name: str | uuid.UUID) -> list[Entity | None]:
        """
        Get a child :obj:`~geoh5py.data.data.Data` by name.

        :param name: Name of the target child data
        :param entity_type: Sub-select entities based on type.
        :return: A list of children Data objects
        """

        if isinstance(name, uuid.UUID):
            entity_list = [child for child in self.children if child.uid == name]
        else:
            entity_list = [child for child in self.children if child.name == name]

        if not entity_list:
            return [None]

        return entity_list

    def get_entity_list(self, entity_type=ABC) -> list[str]:
        """
        Get a list of names of all children :obj:`~geoh5py.data.data.Data`.

        :param entity_type: Option to sub-select based on type.
        :return: List of names of data associated with the object.
        """
        name_list = [
            child.name for child in self.children if isinstance(child, entity_type)
        ]
        return sorted(name_list)

    def remove_children(self, children: list[shared.Entity | PropertyGroup]):
        """
        Remove children from the list of children entities.

        :param children: List of entities

        .. warning::
            Removing a child entity without re-assigning it to a different
            parent may cause it to become inactive. Inactive entities are removed
            from the workspace by
            :func:`~geoh5py.shared.weakref_utils.remove_none_referents`.
        """
        if not isinstance(children, list):
            children = [children]

        self._children = [child for child in self._children if child not in children]
        self.workspace.remove_children(self, children)
