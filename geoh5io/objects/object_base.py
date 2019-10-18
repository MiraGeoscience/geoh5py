import uuid
from abc import abstractmethod
from typing import TYPE_CHECKING, List

from geoh5io.shared import Entity

from .object_type import ObjectType

if TYPE_CHECKING:
    from geoh5io import workspace


class ObjectBase(Entity):
    def __init__(self, object_type: ObjectType, name: str, uid: uuid.UUID = None):
        assert object_type is not None
        super().__init__(name, uid)

        self._type = object_type
        self._allow_move = 1
        self._clipping_ids: List[uuid.UUID] = []
        self._parent = None
        object_type.workspace._register_object(self)

    @property
    def parent(self):
        """
        The parent of an object in the workspace
        :return: Entity: Parent entity
        """
        if getattr(self, "_parent", None) is None:

            self._parent = self.get_parent()

        return self._parent

    def get_parent(self):
        """
        Function to fetch the parent of an object from the workspace tree
        :return: Entity: Parent entity of object
        """

        return self.entity_type.workspace.get_parent(self.uid)[0]

    @property
    def get_data_list(self):
        """
        :return: List of names of data
        """
        return self.entity_type.workspace.get_children_list(self.uid)

    def get_data(self, name: str):
        """
        :return: List of names of data
        """
        return self.entity_type.workspace.get_children(self.uid, name)

    @property
    def entity_type(self) -> ObjectType:
        return self._type

    @classmethod
    def find_or_create_type(cls, workspace: "workspace.Workspace") -> ObjectType:
        return ObjectType.find_or_create(workspace, cls)

    @classmethod
    @abstractmethod
    def default_type_uid(cls) -> uuid.UUID:
        ...
