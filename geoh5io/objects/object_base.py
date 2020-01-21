import uuid
from abc import abstractmethod
from typing import TYPE_CHECKING, List, Union

from numpy import ndarray

from geoh5io.data import Data
from geoh5io.groups import PropertyGroup
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

        # self.has_property_groups = False
        self._property_groups: List[PropertyGroup] = []

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
        name_list = []
        for child in self.children:
            if isinstance(child, Data):
                name_list.append(child.name)
        return name_list

    @property
    def entity_type(self) -> ObjectType:
        return self.__type

    @property
    def property_groups(self) -> List[PropertyGroup]:

        return self._property_groups

    @property_groups.setter
    def property_groups(self, prop_groups: List[PropertyGroup]):

        # First time start with an empty list
        if self.existing_h5_entity:
            self.update_h5 = ["property_groups"]
        self._property_groups = self.property_groups + prop_groups

    @classmethod
    def find_or_create_type(cls, workspace: "workspace.Workspace") -> ObjectType:
        return ObjectType.find_or_create(workspace, cls)

    @classmethod
    @abstractmethod
    def default_type_uid(cls) -> uuid.UUID:
        ...

    @classmethod
    def create(cls, workspace: "workspace.Workspace", save_on_creation=True, **kwargs):
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
            try:
                setattr(new_object, attr, item)
            except AttributeError:
                pass

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

        new_object.parent = parent

        if "data" in kwargs.keys():
            data_objects = new_object.add_data(kwargs["data"])

            return tuple([new_object] + data_objects)

        if save_on_creation:
            workspace.save_entity(new_object)

        return new_object

    def add_property_group(self, groups: dict):
        """
        add_property_group(groups)

        Create property groups from given group names and properties.
        e.g.:
            groups = {
                "MyGroupName1": {"properties":[data1.uid, data2.uid,...],
                "MyGroupName2": {"properties":[data3.uid, data4.uid,...}
            }
        All given data.uid must be children of the object.

        Parameter
        ---------
        groups: dict
            Dictionary of arguments supported by the geoh5io.PropertyGroup object.

        """

        pg_list = []
        for name, attrs in groups.items():

            assert (
                "properties" in attrs.keys()
            ), "Each group must at least have 'properties' assigned"
            children_uid = [child.uid for child in self.children]

            new_pg = PropertyGroup()
            new_pg.group_name = name
            for uid in attrs["properties"]:
                assert isinstance(uid, uuid.UUID), "Given uid must be of type uuid.UUID"
                assert (
                    uid in children_uid
                ), f"Given uid {uid} in group {name} does not match any known children"

            for attr, val in attrs.items():
                try:
                    setattr(new_pg, attr, val)
                except AttributeError:
                    pass

            pg_list.append(new_pg)

        self.property_groups = pg_list

    def get_property_group(
        self, group_id: Union[str, uuid.UUID]
    ) -> List[PropertyGroup]:
        """
        get_property_group(group_name)

        Retrieve a property_group from one of its identifier, either by group_name or uuid

        Parameters
        ----------
        group_id: str | uuid.UUID
            PropertyGroup identifier, either group_name or uuid

        Returns
        -------
        object_list: List[PropertyGroup]
            List of PropertyGroup with the same given group_name
        """

        if isinstance(group_id, uuid.UUID):
            groups_list = [pg for pg in self.property_groups if pg.uid == group_id]

        else:  # Extract all groups uuid with matching group_id
            groups_list = [
                pg for pg in self.property_groups if pg.group_name == group_id
            ]

        return groups_list

    def add_data(self, data: dict, save_on_creation=True):
        """
        add_data(data)

        Create data with association


        Returns
        -------
        data: list[Data]
            List of created Data objects
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

                # Add parent-child relationship
                data_object.parent = self

                if save_on_creation:
                    self.workspace.save_entity(data_object)

                data_objects.append(data_object)

        return data_objects

    def get_data(self, name: str) -> List[Data]:
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
        entity_list = []

        for child in self.children:
            if isinstance(child, Data) and child.name == name:
                entity_list.append(child)

        return entity_list
