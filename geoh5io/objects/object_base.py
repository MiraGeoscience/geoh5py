import uuid
from abc import abstractmethod
from typing import TYPE_CHECKING, List, Optional, Union

from numpy import ndarray

from geoh5io.data import Data
from geoh5io.groups import PropertyGroup
from geoh5io.shared import Entity

from .object_type import ObjectType

if TYPE_CHECKING:
    from geoh5io import workspace


class ObjectBase(Entity):
    """
    Base class object.
    """

    _attribute_map = Entity._attribute_map.copy()
    _attribute_map.update(
        {"Last focus": "last_focus", "PropertyGroups": "property_groups"}
    )

    def __init__(self, object_type: ObjectType, **kwargs):
        assert object_type is not None
        self._type = object_type
        self._property_groups: List[PropertyGroup] = []
        self._last_focus = "None"

        # self._clipping_ids: List[uuid.UUID] = []

        super().__init__(**kwargs)

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
    def find_or_create_type(
        cls, workspace: "workspace.Workspace", **kwargs
    ) -> ObjectType:
        """
        Find or create a type for a given object class

        :param Current workspace: Workspace

        :return: A new or existing object type
        """
        return ObjectType.find_or_create(workspace, cls, **kwargs)

    @classmethod
    @abstractmethod
    def default_type_uid(cls) -> uuid.UUID:
        ...

    def add_data_to_group(self, data: Union[Data, uuid.UUID, str], group_name: str):
        """
        Append data to a property group where the data can be a Data object, its name
        or uid. The given group identifier (name or uid) is created if it does not exist already.
        All given data must be children of the object.

        :param data: Data object or uuid of data
        :param group_name: Name of a property group. A new group is created
        if none exist with the given name.
        """
        prop_group = self.get_property_group(group_name)

        if prop_group is None:
            prop_group = self.create_property_group(group_name=group_name)

        if isinstance(data, Data):
            uid = [data.uid]
        elif isinstance(data, str):
            uid = [obj.uid for obj in self.workspace.get_entity(data)]
        else:
            uid = [data]

        for i in uid:
            assert i in [
                child.uid for child in self.children
            ], f"Given data with uuid {i} does not match any known children"

        prop_group.properties = uid
        self.update_h5 = "property_groups"

    def create_property_group(self, **kwargs) -> PropertyGroup:
        """
        Create property groups from given group names and properties.
        An existing property_group is returned if one exists with the same name.

        :param group_name: Name given to the new PropertyGroup object.

        :return: A new or existing property_group object
        """
        prop_group = PropertyGroup(**kwargs)

        self.property_groups = [prop_group]

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
        self, data: dict, property_group: str = None
    ) -> Union[Data, List[Data]]:
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
                    entity={
                        "name": key,
                        "uid": uuid.uuid4(),
                        "parent": self,
                        "association": value[0].upper(),
                        "values": value[1],
                    },
                    entity_type={"primitive_type": "FLOAT"},
                )

                if property_group is not None:
                    self.add_data_to_group(data_object, property_group)

                data_objects.append(data_object)

        if len(data_objects) == 1:
            return data_object

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
