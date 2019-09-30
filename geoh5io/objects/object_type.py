from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional, Type

from geoh5io.shared import EntityType

if TYPE_CHECKING:
    from geoh5io import workspace
    from . import object_base  # noqa: F401


class ObjectType(EntityType):
    def __init__(
        self,
        workspace: "workspace.Workspace",
        uid: uuid.UUID,
        legacy_class_id: uuid.UUID = None,
    ):
        super().__init__(workspace, uid)
        self._class_id = legacy_class_id

    @staticmethod
    def _is_abstract() -> bool:
        return False

    @property
    def class_id(self) -> Optional[uuid.UUID]:
        """ From legacy file format. Should not need this. """
        return self._class_id

    @classmethod
    def find_or_create(
        cls,
        workspace: "workspace.Workspace",
        object_class: Type["object_base.ObjectBase"],
    ) -> ObjectType:
        """ Find or creates the ObjectType with the pre-defined type UUID that matches the given
        Object implementation class.

        It is expected to have a single instance of ObjectType in the Workspace
        for each concrete Object class.

        :param object_class: An Object implementation class.
        :return: A new instance of ObjectType.
        """
        type_uid = object_class.default_type_uid()
        if type_uid is None:
            raise RuntimeError(
                f"Cannot create GroupType with null UUID from {object_class.__name__}."
            )

        object_type = cls.find(workspace, type_uid)
        if object_type is not None:
            return object_type

        return cls(workspace, type_uid)

    @staticmethod
    def create_custom(workspace: "workspace.Workspace") -> ObjectType:
        """ Creates a new instance of ObjectType for an unlisted custom Object type with a
        new auto-generated UUID.
        """
        type_uid = uuid.uuid4()
        return ObjectType(workspace, type_uid)
