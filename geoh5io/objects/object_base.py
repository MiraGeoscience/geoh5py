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
    def __init__(
        self, object_type: ObjectType, name: str, uid: uuid.UUID = None, parent=None
    ):
        assert object_type is not None
        super().__init__(name, uid)

        self._type = object_type
        self._allow_move = 1
        # self._clipping_ids: List[uuid.UUID] = []
        self._parent = parent

        object_type.workspace._register_object(self)

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
        return self.entity_type.workspace.get_names_of_type(self.uid, "data")

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

            vertices: numpy.ndarray("float64")
                Coordinate xyz vertices [n x 3]

            name: str optional
                Name of the point object [NewPoints]

            uid: uuid.UUID optional
                Unique identifier, or randomly generated using uuid.uuid4 if None

            parent: uuid.UUID | Entity | None optional
                Parental Entity or reference uuid to be linked to.
                If None, the object is added to the base Workspace.

            data: Dict{'name': ['association', values]} optional
                Dictionary of data values associated to "CELL" or "VERTEX" values

            cell:
        Returns
        -------
        entity: geoh5io.Points
            Point object registered to the workspace.
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

        # Add the new object and type to tree
        object_type.workspace.add_to_tree(new_object)

        if "parent" in kwargs.keys():
            new_object.parent = kwargs["parent"]

        for attr in new_object.__dict__:

            if attr[1:] in kwargs.keys():
                setattr(new_object, attr[1:], kwargs[attr[1:]])

        # if hasattr(cls, "_vertices") and "vertices" in kwargs.keys():
        #     assert (
        #         kwargs["vertices"].shape[1] == 3
        #     ), "'vertices' should be an an array of shape N x 3"
        #     new_object.vertices = kwargs["vertices"]
        #
        # if hasattr(cls, "_cells") and "cells" in kwargs.keys():
        #     assert (
        #         kwargs["cells"].shape[1] == 2
        #     ), "'cells' should be an an array of shape N x 2"
        #     new_object.cells = kwargs["cells"]

        if "data" in kwargs.keys():
            data_objects = []
            for key, value in kwargs["data"].items():

                if isinstance(value[1], ndarray):
                    data_object = object_type.workspace.create_entity(
                        Data,
                        key,
                        uuid.uuid4(),
                        entity_type_uid=uuid.uuid4(),
                        attributes={"association": value[0].upper()},
                        type_attributes={"primitive_type": "FLOAT"},
                    )

                    data_object.values = value[1]

                    # Add the new object and type to tree
                    object_type.workspace.add_to_tree(data_object)

                    data_object.parent = new_object

                    data_objects.append(data_object)

            return tuple([new_object] + data_objects)

        return new_object
