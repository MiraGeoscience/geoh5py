import uuid
from abc import abstractmethod

from geoh5io.shared import Entity

from .data_association_enum import DataAssociationEnum
from .data_type import DataType
from .primitive_type_enum import PrimitiveTypeEnum


class Data(Entity):

    attribute_map = Entity.attribute_map.copy()
    attribute_map.update({"Association": "association"})

    def __init__(
        self,
        data_type: DataType,
        association: DataAssociationEnum,
        name: str,
        uid: uuid.UUID = None,
    ):
        assert data_type is not None
        assert data_type.primitive_type == self.primitive_type()
        super().__init__(name, uid)
        self._association = association
        self._type = data_type
        self._values = None
        data_type.workspace._register_data(self)

    @property
    def association(self) -> DataAssociationEnum:
        return self._association

    @association.setter
    def association(self, value):
        if self._association is None:

            assert isinstance(
                value, DataAssociationEnum
            ), f"Association must be of type {DataAssociationEnum}"
            self._association = value

    @property
    def entity_type(self) -> DataType:
        return self._type

    @classmethod
    @abstractmethod
    def primitive_type(cls) -> PrimitiveTypeEnum:
        ...
