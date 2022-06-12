#  Copyright (c) 2022 Mira Geoscience Ltd.
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
import weakref
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar, cast

if TYPE_CHECKING:
    from .. import workspace as ws


TEntityType = TypeVar("TEntityType", bound="EntityType")


class EntityType(ABC):

    _attribute_map = {"Description": "description", "ID": "uid", "Name": "name"}

    def __init__(self, workspace: ws.Workspace, uid: uuid.UUID | None = None, **kwargs):
        assert workspace is not None
        self._workspace = weakref.ref(workspace)

        assert uid is None or isinstance(uid, uuid.UUID)
        self._description: str | None = "Entity"
        self._concatenation = False
        self._name: str | None = "Entity"
        self._on_file = False
        self._uid: uuid.UUID = uid if uid is not None else uuid.uuid4()

        for attr, item in kwargs.items():
            try:
                if attr in self._attribute_map:
                    attr = self._attribute_map[attr]
                setattr(self, attr, item)
            except AttributeError:
                continue

    @property
    def attribute_map(self):
        """
        :obj:`dict` Correspondence map between property names used in geoh5py and
        geoh5.
        """
        return self._attribute_map

    @property
    def concatenation(self):
        """Store the entity as Concatenated, Concatenator or standalone."""
        return self._concatenation

    @property
    def description(self) -> str | None:
        return self._description

    @description.setter
    def description(self, description: str):
        self._description = description
        self.workspace.update_attribute(self, "attributes")

    @property
    def on_file(self) -> bool:
        """
        :obj:`bool` Entity already present in
        :obj:`~geoh5py.workspace.workspace.Workspace.h5file`.
        """
        return self._on_file

    @on_file.setter
    def on_file(self, value: bool):
        self._on_file = value

    @classmethod
    def find(
        cls: type[TEntityType], workspace: ws.Workspace, type_uid: uuid.UUID
    ) -> TEntityType | None:
        """Finds in the given Workspace the EntityType with the given UUID for
        this specific EntityType implementation class.

        :return: EntityType of None
        """
        return cast(TEntityType, workspace.find_type(type_uid, cls))

    @staticmethod
    @abstractmethod
    def _is_abstract() -> bool:
        """Trick to prevent from instantiating abstract base class."""
        return True

    @property
    def name(self) -> str | None:
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name
        self.workspace.update_attribute(self, "attributes")

    @property
    def uid(self) -> uuid.UUID:
        """
        :obj:`uuid.UUID` The unique identifier of an entity, either as stored
        in geoh5 or generated in :func:`~uuid.UUID.uuid4` format.
        """
        return self._uid

    @uid.setter
    def uid(self, uid: str | uuid.UUID):
        if isinstance(uid, str):
            uid = uuid.UUID(uid)

        self._uid = uid
        self.workspace.update_attribute(self, "attributes")

    @property
    def workspace(self) -> ws.Workspace:
        """
        :obj:`~geoh5py.workspace.workspace.Workspace` registering this type.
        """
        workspace = self._workspace()

        # Workspace should never be null, unless this is a dangling type object,
        # which means workspace has been deleted.
        assert workspace is not None
        return workspace
