from .data import Data
from .data import PrimitiveTypeEnum


class FloatData(Data):
    @classmethod
    def primitive_type(cls) -> PrimitiveTypeEnum:
        return PrimitiveTypeEnum.FLOAT

    # TODO: implement specialization to access values.
    # Stored as a 1D array of 32-bit float type.
    # No data value: 1.175494351e-38 (FLT_MIN, considering use of NaN)
