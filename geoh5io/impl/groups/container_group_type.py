from __future__ import annotations

import uuid

from . import GroupType


class ContainerGroupType(GroupType):
    """ The type for the basic Container group."""

    __class_id = uuid.UUID("{61FBB4E8-A480-11E3-8D5A-2776BDF4F982}")

    def __init__(self, uid, name=None, description=None, class_id=None):
        super().__init__(uid, name, description, class_id)

    @classmethod
    def static_class_id(cls) -> uuid.UUID:
        return cls.__class_id

    @classmethod
    def create(cls) -> ContainerGroupType:
        """ Creates a new instance of ContainerGroupType with the UUID dedicated to this class.

        The same UUID is used for class_id. All created instances of ContainerGroupType share the same UUID.
        It is actually expected to have a single instance of this class in a Workspace.
        """
        return ContainerGroupType(
            cls.__class_id, "Container", "Container", cls.__class_id
        )
