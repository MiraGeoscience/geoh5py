import uuid

from geoh5io.data import Data
from geoh5io.data import DataAssociationEnum
from geoh5io.data import DataType
from geoh5io.data import PrimitiveTypeEnum


class UnknownData(Data):
    def __init__(
        self,
        data_type: DataType,
        association: DataAssociationEnum,
        uid: uuid.UUID,
        name: str,
    ):
        super().__init__(data_type, association, uid, name)
        raise NotImplementedError("No implementation for UnknownData")
        # Possibly, provide an implementation to access generic data,
        # for which primitive type is provided by the H5 file.

    @classmethod
    def primitive_type(cls) -> PrimitiveTypeEnum:
        return PrimitiveTypeEnum.UNKNOWN
