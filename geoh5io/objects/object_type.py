from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Type

from geoh5io.shared import EntityType

if TYPE_CHECKING:
    from geoh5io import workspace
    from . import object


class ObjectType(EntityType):
    def __init__(
        self,
        workspace: "workspace.Workspace",
        uid: uuid.UUID,
        class_id: uuid.UUID = None,
    ):
        super().__init__(workspace, uid)
        self._class_id = class_id

    @staticmethod
    def _is_abstract() -> bool:
        return False

    @property
    def class_id(self) -> uuid.UUID:
        """ If class ID was not set, defaults to this type UUID."""
        return self._class_id if self._class_id is not None else self.uid

    @classmethod
    def find_or_create(
        cls, workspace: "workspace.Workspace", object_class: Type["object.Object"]
    ) -> ObjectType:
        """ Find or creates the ObjectType with the class_id from the given Object
        implementation class.

        The class_id is also used as the UUID for the newly created ObjectType.
        It is expected to have a single instance of ObjectType in the Workspace
        for each concrete Object class.

        :param object_class: An Object implementation class.
        :return: A new instance of ObjectType.
        """
        type_uid = object_class.static_type_uid()
        if type_uid is None:
            raise RuntimeError(
                f"Cannot create GroupType with null UUID from {object_class.__name__}."
            )

        object_type = cls.find(workspace, type_uid)
        if object_type is not None:
            return object_type

        class_id = object_class.static_class_id()
        return cls(workspace, type_uid, class_id)

    @staticmethod
    def create_custom(workspace: "workspace.Workspace") -> ObjectType:
        """ Creates a new instance of ObjectType for an unlisted custom Object type with a
        new auto-generated UUID.

        The same UUID is used for class_id.
        """
        class_id = uuid.uuid4()
        return ObjectType(workspace, class_id, class_id)
