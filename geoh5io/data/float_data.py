from .data import Data, PrimitiveTypeEnum


class FloatData(Data):
    @classmethod
    def primitive_type(cls) -> PrimitiveTypeEnum:
        return PrimitiveTypeEnum.FLOAT

    # TODO: implement specialization to access values.
    # Stored as a 1D array of 32-bit float type.
    # No data value: 1.175494351e-38 (FLT_MIN, considering use of NaN)

    @property
    def values(self):
        if getattr(self, "_values", None) is None:
            self._values = self.workspace.fetch_values(self.uid)

        return self._values

    @values.setter
    def values(self, values):

        self._values = values
