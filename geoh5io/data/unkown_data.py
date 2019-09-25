import uuid

from .data import Data
from .data_association_enum import DataAssociationEnum
from .data_type import DataType
from .primitive_type_enum import PrimitiveTypeEnum


class UnknownData(Data):
    def __init__(
        self,
        data_type: DataType,
        association: DataAssociationEnum,
        uid: uuid.UUID,
        name: str,
    ):
        super().__init__(data_type, association, name, uid)
        raise NotImplementedError("No implementation for UnknownData")
        # Possibly, provide an implementation to access generic data,
        # for which primitive type is provided by the H5 file.

    @classmethod
    def primitive_type(cls) -> PrimitiveTypeEnum:
        return PrimitiveTypeEnum.UNKNOWN
