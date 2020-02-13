from numpy import ndarray, ravel

from .data import Data, PrimitiveTypeEnum


class FloatData(Data):
    """
    Data container for floats values
    """

    @classmethod
    def primitive_type(cls) -> PrimitiveTypeEnum:
        return PrimitiveTypeEnum.FLOAT

    @property
    def values(self) -> ndarray:
        """
        :return: values: An array of float values
        """
        if (getattr(self, "_values", None) is None) and self.existing_h5_entity:
            self._values = self.workspace.fetch_values(self.uid)

        return self._values

    @values.setter
    def values(self, values):
        self.modified_attributes = "values"
        self._values = ravel(values)

    def __call__(self):
        return self.values
