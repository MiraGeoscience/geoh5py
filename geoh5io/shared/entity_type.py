from __future__ import annotations

import uuid
import weakref
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Type, TypeVar, Union, cast

if TYPE_CHECKING:
    from geoh5io import workspace as ws


TEntityType = TypeVar("TEntityType", bound="EntityType")


class EntityType(ABC):

    _attribute_map = {"Description": "description", "ID": "uid", "Name": "name"}

    def __init__(self, workspace: "ws.Workspace", **kwargs):
        assert workspace is not None
        self._workspace = weakref.ref(workspace)

        self._uid: uuid.UUID = uuid.uuid4()
        self._name = "None"
        self._description: Optional[str] = None
        self._existing_h5_entity = False
        self._update_h5 = False

        for attr, item in kwargs.items():
            try:
                if attr in self._attribute_map.keys():
                    attr = self._attribute_map[attr]
                setattr(self, attr, item)
            except AttributeError:
                continue

    @property
    def attribute_map(self):
        return self._attribute_map

    @property
    def existing_h5_entity(self) -> bool:
        return self._existing_h5_entity

    @existing_h5_entity.setter
    def existing_h5_entity(self, value: bool):
        self._existing_h5_entity = value

    @property
    def update_h5(self) -> bool:
        return self._update_h5

    @update_h5.setter
    def update_h5(self, value: bool):
        self._update_h5 = value

    @staticmethod
    @abstractmethod
    def _is_abstract() -> bool:
        """ Trick to prevent from instantiating abstract base class. """
        return True

    @property
    def workspace(self) -> "ws.Workspace":
        """ Return the workspace which owns this type. """
        workspace = self._workspace()

        # Workspace should never be null, unless this is a dangling type object,
        # which workspace has been deleted.
        assert workspace is not None
        return workspace

    @property
    def uid(self) -> uuid.UUID:
        return self._uid

    @uid.setter
    def uid(self, uid: Union[str, uuid.UUID]):
        if isinstance(uid, str):
            uid = uuid.UUID(uid)

        self._uid = uid

    @property
    def name(self) -> Optional[str]:
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    @property
    def description(self) -> Optional[str]:
        return self._description

    @description.setter
    def description(self, description: str):
        self._description = description

    @description.getter
    def description(self):

        if self._description is None:
            self.description = self.name

        return self._description

    @classmethod
    def find(
        cls: Type[TEntityType], workspace: "ws.Workspace", type_uid: uuid.UUID
    ) -> Optional[TEntityType]:
        """ Finds in the given Workspace the EntityType with the given UUID for
        this specific EntityType implementation class.

        Returns None if not found.
        """
        return cast(TEntityType, workspace.find_type(type_uid, cls))
