from __future__ import annotations

import uuid
import weakref
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Type, TypeVar, cast

if TYPE_CHECKING:
    from geoh5io import workspace as ws

TEntityType = TypeVar("TEntityType", bound="EntityType")


class EntityType(ABC):
    def __init__(
        self,
        workspace: "ws.Workspace",
        uid: uuid.UUID,
        name: str = None,
        description: str = None,
    ):
        assert workspace is not None
        assert uid is not None
        assert uid.int != 0

        self._workspace = weakref.ref(workspace)
        self._uid = uid
        self._name = name
        self._description = description
        workspace.register_type(self)

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

    @property
    def name(self) -> Optional[str]:
        return self._name

    @property
    def description(self) -> Optional[str]:
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

    @classmethod
    def known_types(cls):
        """
        Get dictionary of known entity types with class

        :return: dict
        """
        return {
            "{dd99b610-be92-48c0-873c-5b5946ea2840}": "Unknown type",
            "{61fbb4e8-a480-11e3-8d5a-2776bdf4f982}": "Container",
            "{825424fb-c2c6-4fea-9f2b-6cd00023d393}": "DrillHole container",
            "{202c5db1-a56d-4004-9cad-baafd8899406}": "Points",
            "{6a057fdc-b355-11e3-95be-fd84a7ffcb88}": "Curve",
            "{f26feba3-aded-494b-b9e9-b2bbcbe298e1}": "Surface",
            "{b020a277-90e2-4cd7-84d6-612ee3f25051}": "3D mesh (block model)",
            "{7caebf0e-d16e-11e3-bc69-e4632694aa37}": "Drillhole",
            "{77ac043c-fe8d-4d14-8167-75e300fb835a}": "GeoImage",
            "{e79f449d-74e3-4598-9c9c-351a28b8b69e}": "Label",
        }
