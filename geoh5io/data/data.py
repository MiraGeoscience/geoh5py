from abc import abstractmethod
from typing import TYPE_CHECKING, Optional, Type

from geoh5io.shared import Entity

from .data_association_enum import DataAssociationEnum
from .data_type import DataType
from .primitive_type_enum import PrimitiveTypeEnum

if TYPE_CHECKING:
    from geoh5io import workspace


class Data(Entity):

    _attribute_map = Entity._attribute_map.copy()
    _attribute_map.update({"Association": "association"})

    def __init__(self, data_type: DataType, **kwargs):
        assert data_type is not None
        assert data_type.primitive_type == self.primitive_type()

        self._type = data_type
        self._association: Optional[DataAssociationEnum] = None
        self._values = None
        super().__init__(**kwargs)

        data_type.workspace._register_data(self)

    @property
    def association(self) -> Optional[DataAssociationEnum]:
        return self._association

    @association.setter
    def association(self, value):
        if isinstance(value, str):
            self._association = getattr(DataAssociationEnum, value.upper())
        else:
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

    @classmethod
    def find_or_create_type(
        cls: Type[Entity], workspace: "workspace.Workspace", **kwargs
    ) -> DataType:
        """
        Find or create a type for a given object class

        :param Current workspace: Workspace

        :return: A new or existing object type
        """
        return DataType.find_or_create(workspace, **kwargs)
