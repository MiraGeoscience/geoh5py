import uuid
from abc import abstractmethod
from typing import TYPE_CHECKING, Optional

from geoh5io.shared import Entity

from .group_type import GroupType

if TYPE_CHECKING:
    from geoh5io import workspace


class Group(Entity):
    def __init__(self, group_type: GroupType, name: str, uid: uuid.UUID = None):
        assert group_type is not None
        super().__init__(name, uid)

        self._type = group_type
        self._allow_move = True

        # self._clipping_ids: List[uuid.UUID] = []
        group_type.workspace._register_group(self)

    @property
    def entity_type(self) -> GroupType:
        return self._type

    @classmethod
    def find_or_create_type(cls, workspace: "workspace.Workspace") -> GroupType:
        return GroupType.find_or_create(workspace, cls)

    @property
    def allow_move(self) -> bool:
        return self._allow_move

    @allow_move.setter
    def allow_move(self, value: bool):
        self._allow_move = value

    @classmethod
    @abstractmethod
    def default_type_uid(cls) -> Optional[uuid.UUID]:
        ...

    @classmethod
    @abstractmethod
    def default_type_name(cls) -> Optional[str]:
        ...

    @classmethod
    def default_type_description(cls) -> Optional[str]:
        return cls.default_type_name()

    @classmethod
    def create(cls, workspace: "workspace.Workspace", save_on_creation=True, **kwargs):
        """
        create(
            workspace, name=["NewGroup"],
            uid=[uuid.uuid4()], parent=[None]
        )

        Function to create a group object

        Parameters
        ----------
        workspace: geoh5io.Workspace
            Workspace to be added to

        **kwargs
            name: str optional
                Name of the Group object ["NewGroup"]

            uid: uuid.UUID optional
                Unique identifier, or randomly generated using uuid.uuid4 if None

            parent: uuid.UUID | Entity | None optional
                Parental Entity or reference uuid to be linked to.
                If None, the object is added to the base Workspace.

        Returns
        -------
        entity: geoh5io.Group
            Group object registered to the workspace.
        """
        new_group_type = cls.find_or_create_type(workspace)

        if "name" in kwargs.keys():
            name = kwargs["name"]
        else:
            name = "NewGroup"

        if "uid" in kwargs.keys():
            assert isinstance(
                kwargs["uid"], uuid.UUID
            ), "Input uid must be of type uuid.UUID"
            uid = kwargs["uid"]
        else:
            uid = uuid.uuid4()

        new_group = cls(new_group_type, name, uid)

        # Replace all attributes given as kwargs
        for attr, item in kwargs.items():
            try:
                setattr(new_group, attr, item)
            except AttributeError:
                pass  # print(f"Could not set attribute {attr}")

        # Add parent-child relationship
        if "parent" in kwargs.keys():
            if isinstance(kwargs["parent"], uuid.UUID):
                parent = workspace.get_entity(kwargs["parent"])[0]
            else:
                assert isinstance(
                    kwargs["parent"], Entity
                ), "Given 'parent' argument must be of type uuid.UUID or 'Entity'"

                parent = kwargs["parent"]
        else:
            parent = workspace.root

        new_group.parent = parent

        if save_on_creation:
            workspace.save_entity(new_group)

        return new_group
