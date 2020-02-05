import uuid
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

from numpy import ndarray

from geoh5io.data import Data
from geoh5io.groups import PropertyGroup
from geoh5io.shared import Entity

from .object_type import ObjectType

if TYPE_CHECKING:
    from geoh5io import workspace


class ObjectBase(Entity):
    """
    Base class for objects.
    """

    _attribute_map = Entity._attribute_map.copy()
    _attribute_map.update(
        {"Last focus": "last_focus", "PropertyGroups": "property_groups"}
    )

    def __init__(self, object_type: ObjectType, name: str, uid: uuid.UUID = None):
        assert object_type is not None
        super().__init__(name, uid)

        self._type = object_type
        # self._clipping_ids: List[uuid.UUID] = []
        self._property_groups: List[PropertyGroup] = []
        self._last_focus = "None"

        object_type.workspace._register_object(self)

    @property
    def last_focus(self) -> str:
        """
        Object visible in camera on start: bool
        """
        return self._last_focus

    @last_focus.setter
    def last_focus(self, value: str):
        self._last_focus = value

    @property
    def entity_type(self) -> ObjectType:
        """
        Object type: EntityType
        """
        return self._type

    @property
    def property_groups(self) -> List[PropertyGroup]:
        """
        List of property (data) groups: List[PropertyGroup]
        """
        return self._property_groups

    @property_groups.setter
    def property_groups(self, prop_groups: List[PropertyGroup]):

        for prop_group in prop_groups:
            prop_group.parent = self

        # First time start with an empty list
        self.update_h5 = "property_groups"
        self._property_groups = self.property_groups + prop_groups

    @classmethod
    def find_or_create_type(cls, workspace: "workspace.Workspace") -> ObjectType:
        """
        Find or create a type for a given object class

        :param Current workspace: Workspace

        :return: A new or existing object type
        """
        return ObjectType.find_or_create(workspace, cls)

    @classmethod
    @abstractmethod
    def default_type_uid(cls) -> uuid.UUID:
        ...

    @classmethod
    def create(
        cls, workspace: "workspace.Workspace", **kwargs
    ) -> Union[Entity, Tuple[Any, ...]]:
        """
        Function to create an object with data

        :param workspace: Workspace to be added to
        :param kwargs: List of keyword arguments

        :return: Entity registered to the workspace.
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

            return tuple([new_object, data_objects])

        return new_object

    def add_data_to_group(
        self, data: Union[Data, uuid.UUID, str], group_id: Union[str, uuid.UUID]
    ):
        """
        Append data to a property group where the data can be a Data object or a uuid
        e.g.:
            data = [FloatData, data.uid]

        All given data must be children of the object.
        The given group_id name is created if it does not exist already.

        :param data: Data object or uuid of data
        :param group_id: PropertyGroup or name of a group. A new group is created
        if none exist with the given name.
        """

        children_uid = [child.uid for child in self.children]

        # Create or retrieve a property_group from the object
        prop_group = self.create_property_group(group_id)

        if isinstance(data, Data):
            uid = data.uid

        assert (
            uid in children_uid
        ), f"Given data with uuid {uid} does not match any known children"

        prop_group.properties = [uid]
        self.update_h5 = "property_groups"

    def create_property_group(
        self, group_name: Union[str, uuid.UUID], **kwargs
    ) -> PropertyGroup:
        """
        Create property groups from given group names and properties.
        An existing property_group is returned if one exists with the same name.

        :param group_name: Name given to the new PropertyGroup object.

        :return: A new or existing property_group object
        """
        # Check if the object has it
        prop_group = self.get_property_group(group_name)

        if prop_group is None:

            if isinstance(group_name, uuid.UUID):
                prop_group = PropertyGroup(uid=group_name)
            else:
                prop_group = PropertyGroup(group_name=group_name)

            self.property_groups = [prop_group]

            for attr, val in kwargs.items():
                try:
                    setattr(prop_group, attr, val)
                except AttributeError:
                    pass

            prop_group.parent = self
            self.update_h5 = "property_groups"

        return prop_group

    def get_property_group(
        self, group_id: Union[str, uuid.UUID]
    ) -> Optional[PropertyGroup]:
        """
        Retrieve a property_group from one of its identifier, either by group_name or uuid

        :param group_id: PropertyGroup identifier, either group_name or uuid

        :return: PropertyGroup with the given group_name
        """

        if isinstance(group_id, uuid.UUID):
            groups_list = [pg for pg in self.property_groups if pg.uid == group_id]

        else:  # Extract all groups uuid with matching group_id
            groups_list = [
                pg for pg in self.property_groups if pg.group_name == group_id
            ]

        if len(groups_list) < 1:
            return None

        return groups_list[0]

    def add_data(
        self, data: dict, property_group: str = None, save_on_creation: bool = True
    ) -> List[Data]:
        """
        Create data with association

        :param data: Dictionary of data to be added to the object, such as
            data = {
            "data_name1": ["CELL", values1],
            "data_name2": ["VERTEX", values2],
            ...
            }

        :return: List of created Data objects data:
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

                if property_group is not None:
                    self.add_data_to_group(data_object, property_group)

                data_objects.append(data_object)

        return data_objects

    def get_data(self, name: str) -> List[Data]:
        """
        Get data objects by name

        :param name: Name of the target child data

        :return: A list of children Data objects
        """
        entity_list = []

        for child in self.children:
            if isinstance(child, Data) and child.name == name:
                entity_list.append(child)

        return entity_list

    def get_data_list(self) -> List[str]:
        """
        :return: List of names of data associated with the object

        """
        name_list = []
        for child in self.children:
            if isinstance(child, Data):
                name_list.append(child.name)
        return name_list
