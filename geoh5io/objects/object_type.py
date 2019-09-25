from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional, Type, cast

from geoh5io.shared import EntityType

if TYPE_CHECKING:
    from geoh5io import workspace
    from . import object


class ObjectType(EntityType):
    def __init__(self, workspace: 'workspace.Workspace', uid: uuid.UUID, class_id: uuid.UUID):
        super().__init__(workspace, uid)
        self._class_id = class_id

    @classmethod
    def find(cls, workspace: 'workspace.Workspace', type_uid: uuid.UUID) -> Optional[ObjectType]:
        return cast(ObjectType, workspace.find_type(type_uid, cls))

    @classmethod
    def find_or_create(cls, workspace: 'workspace.Workspace',
                       object_class: Type["object.Object"])-> ObjectType:
        """ Find or creates the ObjectType with the class_id from the given Object
        implementation class.

        The class_id is also used as the UUID for the newly created ObjectType.
        It is expected to have a single instance of ObjectType in the Workspace
        for each concrete Object class.

        :param object_class: An Object implementation class.
        :return: A new instance of ObjectType.
        """
        class_id = object_class.static_class_id()
        if class_id is None:
            raise RuntimeError(
                f"Cannot create GroupType with null UUID from {object_class.__name__}."
            )

        object_type = cls.find(workspace, class_id)
        if object_type is not None:
            return object_type

        return cls(workspace, class_id, class_id)

    @staticmethod
    def create_custom(workspace: 'workspace.Workspace') -> ObjectType:
        """ Creates a new instance of ObjectType for an unlisted custom Object type with a
        new auto-generated UUID.

        The same UUID is used for class_id.
        """
        class_id = uuid.uuid4()
        return ObjectType(workspace, class_id, class_id)
