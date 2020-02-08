import uuid

from .data import Data
from .data_association_enum import DataAssociationEnum
from .data_type import DataType
from .primitive_type_enum import PrimitiveTypeEnum
from .reference_value_map import ReferenceValueMap


class ReferencedData(Data):
    def __init__(
        self,
        data_type: DataType,
        association: DataAssociationEnum,
        name: str,
        uid: uuid.UUID = None,
    ):
        super().__init__(data_type, association=association, name=name, uid=uid)
        self._value_map = ReferenceValueMap()

    @classmethod
    def primitive_type(cls) -> PrimitiveTypeEnum:
        return PrimitiveTypeEnum.REFERENCED

    # TODO: implement specialization to access values.
    # Stored as a 1D array of UTF-8 encoded, variable-length string type designating a file name.
    # For each file name within "Data", an opaque data set named after the filename must
    # be added under the Data instance, containing a complete binary dump of the file.
    # Different files (under the same object/group) must be saved under different names.
    # No data value : empty string
