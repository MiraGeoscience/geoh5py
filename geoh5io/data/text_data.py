from .data import Data
from .primitive_type_enum import PrimitiveTypeEnum


class TextData(Data):
    @classmethod
    def primitive_type(cls) -> PrimitiveTypeEnum:
        return PrimitiveTypeEnum.TEXT

    # TODO: implement specialization to access values.
    # Stored as a 1D array of UTF-8 encoded, variable-length string type.
    # No data value : empty string
