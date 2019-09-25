from __future__ import annotations

import uuid
from typing import cast
from typing import Optional
from typing import Type
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import object

from geoh5io.shared import EntityType
from geoh5io.workspace import Workspace


class ObjectType(EntityType):
    def __init__(self, uid: uuid.UUID, class_id: uuid.UUID):
        super().__init__(uid)
        self._class_id = class_id

    @classmethod
    def find(cls, type_uid: uuid.UUID) -> Optional[ObjectType]:
        return cast(ObjectType, Workspace.active().find_type(type_uid, cls))

    @classmethod
    def find_or_create(cls, object_class: Type["object.Object"]) -> ObjectType:
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

        object_type = cls.find(class_id)
        if object_type is not None:
            return object_type

        return cls(class_id, class_id)

    @staticmethod
    def create_custom() -> ObjectType:
        """ Creates a new instance of ObjectType for an unlisted custom Object type with a
        new auto-generated UUID.

        The same UUID is used for class_id.
        """
        class_id = uuid.uuid4()
        return ObjectType(class_id, class_id)
