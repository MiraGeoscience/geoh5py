import uuid
from abc import abstractmethod
from typing import TYPE_CHECKING, Optional

from numpy import ndarray

from geoh5io.data import Data
from geoh5io.shared import Entity

from .object_type import ObjectType

if TYPE_CHECKING:
    from geoh5io import workspace


class ObjectBase(Entity):
    def __init__(self, object_type: ObjectType, name: str, uid: uuid.UUID = None):
        assert object_type is not None
        super().__init__(name, uid)

        self.__type = object_type
        self._allow_move = True
        # self._clipping_ids: List[uuid.UUID] = []

        object_type.workspace._register_object(self)

    @property
    def allow_move(self) -> bool:
        return self._allow_move

    @allow_move.setter
    def allow_move(self, value: bool):
        self._allow_move = value

    @property
    def get_data_list(self):
        """
        @property
        get_data_list

        Returns
        -------
        names: list[str]
            List of names of data associated with the object

        """
        return self.workspace.get_names_of_type(self.uid, "data")

    @property
    def entity_type(self) -> ObjectType:
        return self.__type

    @classmethod
    def find_or_create_type(cls, workspace: "workspace.Workspace") -> ObjectType:
        return ObjectType.find_or_create(workspace, cls)

    @classmethod
    @abstractmethod
    def default_type_uid(cls) -> uuid.UUID:
        ...

    @classmethod
    def create(cls, workspace: "workspace.Workspace", **kwargs):
        """
        create(
            locations, workspace=None, name="NewPoints",
            uid=uuid.uuid4(), parent=None, data=None
        )

        Function to create an object with data

        Parameters
        ----------
        workspace: geoh5io.Workspace
            Workspace to be added to

        **kwargs
            List of keyword arguments

        Returns
        -------
        entity: geoh5io.Entity
            Entity registered to the workspace.
        """

        object_type = cls.find_or_create_type(workspace)

        if object_type.name is None:
            object_type.name = cls.__name__

        if object_type.description is None:
            object_type.description = cls.__name__

        if "name" in kwargs.keys():
            name = kwargs["name"]
        else:
            name = "NewObject"

        if "uid" in kwargs.keys():
            assert isinstance(
                kwargs["uid"], uuid.UUID
            ), "Input uid must be of type uuid.UUID"
            uid = kwargs["uid"]
        else:
            uid = uuid.uuid4()

        new_object = cls(object_type, name, uid)

        # Replace all attributes given as kwargs
        for attr, item in kwargs.items():
            if "_" + attr in new_object.__dict__:
                setattr(new_object, attr, item)

        if new_object.parent is not None:
            parent_uid = new_object.parent.uid
        else:
            parent_uid = object_type.workspace.uid

        # Add the new object and type to tree
        object_type.workspace.add_to_tree(new_object, parent=parent_uid)

        if "data" in kwargs.keys():
            data_objects = new_object.add_data(kwargs["data"])

            return tuple([new_object] + data_objects)

        return new_object

    def add_data(self, data: dict):
        """
        add_data(data)

        Create data with association


        :return:
        """

        data_objects = []
        for key, value in data.items():

            if isinstance(value[1], ndarray):
                data_object = self.workspace.create_entity(
                    Data,
                    key,
                    uuid.uuid4(),
                    entity_type_uid=uuid.uuid4(),
                    attributes={"association": value[0].upper()},
                    type_attributes={"primitive_type": "FLOAT"},
                )

                data_object.values = value[1]

                # Add the new object and type to tree
                self.workspace.add_to_tree(data_object)
                data_object.parent = self
                data_objects.append(data_object)

        return data_objects

    def get_data(self, name: str) -> Optional[Entity]:
        """
        @property
        get_data

        Parameters
        ----------
        name: str
            Name of the target child data

        Returns
        -------
        data: geoh5io.Data
            Returns a registered Data
        """
        return self.workspace.get_child(self.uid, name)[0]
