import uuid
from abc import abstractmethod
from typing import TYPE_CHECKING, Optional

from geoh5io.io import H5Writer
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
        # self._clipping_ids: List[uuid.UUID] = []
        # self._parent = None
        object_type.workspace._register_object(self)

    @property
    def get_data_list(self):
        """
        :return: List of names of data
        """
        return self.entity_type.workspace.get_names_of_type(self.uid, "data")

    def get_data(self, name: str) -> Optional[Entity]:
        """
        :return: List of names of data
        """
        return self.entity_type.workspace.get_child(self.uid, name)[0]

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

    def save_to_h5(self, close_file: bool = True):

        H5Writer.save_entity(self, close_file=close_file)
