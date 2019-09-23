from __future__ import annotations

import uuid
from typing import Optional
from typing import Type

from geoh5io.groups import Group
from geoh5io.shared import EntityType


class GroupType(EntityType):
    def __init__(self, uid, name=None, description=None, class_id: uuid.UUID = None):
        super().__init__(uid, name, description)

        self._class_id = class_id
        self._allow_move_content = True
        self._allow_delete_content = True

    @property
    def allow_move_content(self) -> bool:
        return self._allow_move_content

    @allow_move_content.setter
    def allow_move_content(self, allow: bool):
        self._allow_move_content = bool(allow)

    @property
    def allow_delete_content(self) -> bool:
        return self._allow_delete_content

    @allow_delete_content.setter
    def allow_delete_content(self, allow: bool):
        self._allow_delete_content = bool(allow)

    @property
    def class_id(self) -> Optional[uuid.UUID]:
        return self._class_id

    @classmethod
    def create(cls, entity_class: Type[Group]) -> GroupType:
        """ Creates a new instance of GroupType with the class_id from the given Group
        implementation class.

        The class_id is also used as the UUID for the newly created GroupType.
        Thus, all created instances for the same Group class share the same UUID.
        It is actually expected to have a single instance of GroupType in the Workspace
        for each concrete Group class.

        :param entity_class: A Group implementation class.
        :return: A new instance of GroupType.
        """
        assert issubclass(entity_class, Group)
        class_id = entity_class.static_class_id()
        if class_id is None:
            raise RuntimeError(
                f"Cannot create GroupType with null UUID from {entity_class.__name__}."
            )

        return GroupType(
            class_id,
            entity_class.static_type_name(),
            entity_class.static_type_description(),
            class_id,
        )

    @staticmethod
    def create_custom() -> GroupType:
        """ Creates a new instance of GroupType for an unlisted custom Group type with a
        new auto-generated UUID.

        The same UUID is used for class_id.
        """
        class_id = uuid.uuid4()
        return GroupType(class_id, None, None, class_id)
