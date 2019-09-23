from geoh5io.data import Data
from geoh5io.data import PrimitiveTypeEnum


class IntegerData(Data):
    @classmethod
    def primitive_type(cls) -> PrimitiveTypeEnum:
        return PrimitiveTypeEnum.INTEGER

    # TODO: implement specialization to access values.
    # Stored as a 1D array of 32-bit integer type.
    # No data value: –2147483648 (INT_MIN, considering use of NaN)
