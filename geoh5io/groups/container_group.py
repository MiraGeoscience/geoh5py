from __future__ import annotations

import uuid
from typing import Type

from .group import Group
from .group_type import GroupType


class ContainerGroup(Group):
    """ The type for the basic Container group."""

    __class_id = uuid.UUID("{61FBB4E8-A480-11E3-8D5A-2776BDF4F982}")
    __type_name = "Container"

    @classmethod
    def static_class_id(cls) -> uuid.UUID:
        return cls.__class_id

    @classmethod
    def static_type_name(cls) -> str:
        return cls.__type_name

    @classmethod
    def static_type_description(cls) -> str:
        return cls.static_type_name()

    @classmethod
    def find_or_create(cls, group_class: Type[Group]) -> GroupType:
        """ Find or creates the GroupType with the class_id from the given Group
        implementation class.

        The class_id is also used as the UUID for the newly created GroupType.
        It is expected to have a single instance of GroupType in the Workspace
        for each concrete Group class.

        :param group_class: An Group implementation class.
        :return: A new instance of GroupType.
        """
        assert issubclass(group_class, Group)
        class_id = group_class.static_class_id()
        if class_id is None:
            raise RuntimeError(
                f"Cannot create GroupType with null UUID from {group_class.__name__}."
            )

        group_type = cls.find(class_id)
        if group_type is not None:
            return group_type

        return cls(
            class_id,
            group_class.static_type_name(),
            group_class.static_type_description(),
            class_id,
        )
