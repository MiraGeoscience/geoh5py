#  Copyright (c) 2021 Mira Geoscience Ltd.
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
from typing import TYPE_CHECKING, List, Optional, Type, TypeVar, Union, cast

if TYPE_CHECKING:
    from .. import workspace as ws


TEntityType = TypeVar("TEntityType", bound="EntityType")


class EntityType(ABC):

    _attribute_map = {"Description": "description", "ID": "uid", "Name": "name"}

    def __init__(self, workspace: "ws.Workspace", **kwargs):
        assert workspace is not None
        self._workspace = weakref.ref(workspace)

        self._uid: uuid.UUID = uuid.uuid4()
        self._name: Optional[str] = None
        self._description: Optional[str] = None
        self._existing_h5_entity = False
        self._modified_attributes: List[str] = []

        for attr, item in kwargs.items():
            try:
                if attr in self._attribute_map.keys():
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
    def description(self) -> Optional[str]:
        return self._description

    @description.setter
    def description(self, description: str):
        self._description = description
        self.modified_attributes = "attributes"

    @description.getter
    def description(self):
        """
        :obj:`str` Entity type description.
        """
        if self._description is None:
            self.description = self.name

        return self._description

    @property
    def existing_h5_entity(self) -> bool:
        """
        :obj:`bool` Entity already present in
        :obj:`~geoh5py.workspace.workspace.Workspace.h5file`.
        """
        return self._existing_h5_entity

    @existing_h5_entity.setter
    def existing_h5_entity(self, value: bool):
        self._existing_h5_entity = value

    @classmethod
    def find(
        cls: Type[TEntityType], workspace: "ws.Workspace", type_uid: uuid.UUID
    ) -> Optional[TEntityType]:
        """Finds in the given Workspace the EntityType with the given UUID for
        this specific EntityType implementation class.

        :return: EntityType of None
        """
        return cast(TEntityType, workspace.find_type(type_uid, cls))

    @property
    def modified_attributes(self):
        """
        :obj:`list[str]` List of attributes to be updated in associated workspace
        :obj:`~geoh5py.workspace.workspace.Workspace.h5file`.
        """
        return self._modified_attributes

    @modified_attributes.setter
    def modified_attributes(self, values: Union[List, str]):
        if self.existing_h5_entity:
            if not isinstance(values, list):
                values = [values]

            # Check if re-setting the list or appending
            if len(values) == 0:
                self._modified_attributes = []
            else:
                for value in values:
                    if value not in self._modified_attributes:
                        self._modified_attributes.append(value)

    @staticmethod
    @abstractmethod
    def _is_abstract() -> bool:
        """ Trick to prevent from instantiating abstract base class. """
        return True

    @property
    def name(self) -> Optional[str]:
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name
        self.modified_attributes = "attributes"

    @name.getter
    def name(self):
        """
        :obj:`str` Name of the entity type.
        """
        if self._name is None:
            return str(self.uid)

        return self._name

    @property
    def uid(self) -> uuid.UUID:
        """
        :obj:`uuid.UUID` The unique identifier of an entity, either as stored
        in geoh5 or generated in :func:`~uuid.UUID.uuid4` format.
        """
        return self._uid

    @uid.setter
    def uid(self, uid: Union[str, uuid.UUID]):
        if isinstance(uid, str):
            uid = uuid.UUID(uid)
        self.modified_attributes = "attributes"

        self._uid = uid

    @property
    def workspace(self) -> "ws.Workspace":
        """
        :obj:`~geoh5py.workspace.workspace.Workspace` registering this type.
        """
        workspace = self._workspace()

        # Workspace should never be null, unless this is a dangling type object,
        # which means workspace has been deleted.
        assert workspace is not None
        return workspace
