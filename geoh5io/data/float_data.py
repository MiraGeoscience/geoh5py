from numpy import ravel

from .data import Data, PrimitiveTypeEnum


class FloatData(Data):
    @classmethod
    def primitive_type(cls) -> PrimitiveTypeEnum:
        return PrimitiveTypeEnum.FLOAT

    @property
    def values(self):
        if (getattr(self, "_values", None) is None) and self.existing_h5_entity:
            self._values = self.workspace.fetch_values(self.uid)

        return self._values

    @values.setter
    def values(self, values):
        """
        values(values)

        Assign array of float as values

        Parameter
        ---------
        values: numpy.ndarray
            Array of floats

        """
        if self.existing_h5_entity:
            self.update_h5 = "values"

        self._values = ravel(values)

    def __call__(self):
        return self.values
