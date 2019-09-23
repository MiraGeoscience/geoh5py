from __future__ import annotations

import uuid
from typing import Type

from geoh5io.objects import Object
from geoh5io.shared import EntityType


class ObjectType(EntityType):
    def __init__(self, uid: uuid.UUID, class_id: uuid.UUID):
        super().__init__(uid)
        self._class_id = class_id

    @classmethod
    def create(cls, entity_class: Type[Object]) -> ObjectType:
        """ Creates a new instance of ObjectType with the class_id from the given Object
        implementation class.

        The class_id is also used as the UUID for the newly created ObjectType.
        Thus, all created instances for the same Object class share the same UUID.
        It is actually expected to have a single instance of ObjectType in the Workspace
        for each concrete Object class.

        :param entity_class: An Object implementation class.
        :return: A new instance of ObjectType.
        """
        assert issubclass(entity_class, Object)
        class_id = entity_class.static_class_id()
        if class_id is None:
            raise RuntimeError(
                f"Cannot create GroupType with null UUID from {entity_class.__name__}."
            )

        return ObjectType(class_id, class_id)
