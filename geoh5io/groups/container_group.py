import uuid

from geoh5io import workspace

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

    @classmethod
    def create(
        cls,
        work_space=None,
        name: str = "NewPoints",
        uid: uuid.UUID = uuid.uuid4(),
        group_type=None,
        parent=None,
    ):
        if group_type is None:
            group_type = cls.find_or_create_type(
                workspace.Workspace.active() if work_space is None else work_space
            )

        group = ContainerGroup(group_type, name, uid)

        # Add the new object and type to tree
        group_type.workspace.add_to_tree(group)

        group.set_parent(parent)

        return group
