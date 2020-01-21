import uuid
from typing import List, Union

from geoh5io.data import DataAssociationEnum


class PropertyGroup:
    """ Group for properties"""

    def __init__(self, uid: uuid.UUID = uuid.uuid4()):

        self._group_name = "prop_group"
        self._uid = uid
        self._association: DataAssociationEnum = DataAssociationEnum.VERTEX
        self._properties: List[uuid.UUID] = []
        self._property_group_type = "multi-element"

    @property
    def uid(self) -> uuid.UUID:
        return self._uid

    @property
    def group_name(self) -> str:
        return self._group_name

    @group_name.setter
    def group_name(self, new_group_name: str):
        self._group_name = new_group_name

    @property
    def association(self) -> DataAssociationEnum:
        return self._association

    @association.setter
    def association(self, value):
        if self._association is None:

            if isinstance(value, str):
                value = getattr(DataAssociationEnum, value.upper())

            assert isinstance(
                value, DataAssociationEnum
            ), f"Association must be of type {DataAssociationEnum}"
            self._association = value

    @property
    def properties(self) -> List[uuid.UUID]:
        return self._properties

    @properties.setter
    def properties(self, uids: List[Union[str, uuid.UUID]]):

        properties = []
        for uid in uids:
            if isinstance(uid, str):
                uid = uuid.UUID(uid)
            properties.append(uid)
        self._properties = properties

    @property
    def property_group_type(self) -> str:
        return self._property_group_type

    @property_group_type.setter
    def property_group_type(self, group_type: str):
        self._group_type = group_type
