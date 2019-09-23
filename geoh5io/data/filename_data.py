from geoh5io.data import Data
from geoh5io.data import PrimitiveTypeEnum


class FilenameData(Data):
    @classmethod
    def primitive_type(cls) -> PrimitiveTypeEnum:
        return PrimitiveTypeEnum.FILENAME

    # TODO: implement specialization to access values.
    # Stored as a 1D array of 32-bit unsigned integer type (native).
    # Value map : 1D composite type array data set
    #   – Key (unsigned int)
    #   - Value (variable-length utf8 string)
    # must exist under type.
    # No data value : 0 (key is tied to value “Unknown”)
