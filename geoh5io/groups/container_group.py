import uuid

from .group import Group


class ContainerGroup(Group):
    """ The type for the basic Container group."""

    __TYPE_UID = uuid.UUID(
        fields=(0x61FBB4E8, 0xA480, 0x11E3, 0x8D, 0x5A, 0x2776BDF4F982)
    )

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID

    @classmethod
    def default_type_name(cls) -> str:
        return "Container"
