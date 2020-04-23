import numpy as np

from .data import Data, DataType, PrimitiveTypeEnum


class FloatData(Data):
    """
    Data container for floats values
    """

    def __init__(self, data_type: DataType, **kwargs):
        super().__init__(data_type, **kwargs)

    @classmethod
    def primitive_type(cls) -> PrimitiveTypeEnum:
        return PrimitiveTypeEnum.FLOAT

    @property
    def values(self) -> np.ndarray:
        """
        :return: values: An array of float values
        """
        if (getattr(self, "_values", None) is None) and self.existing_h5_entity:
            self._values = self.check_vector_length(
                self.workspace.fetch_values(self.uid)
            )

        return self._values

    @values.setter
    def values(self, values):
        self.modified_attributes = "values"
        self._values = self.check_vector_length(values)

    def check_vector_length(self, values) -> np.ndarray:

        full_vector = np.ones(self.n_values) * self.no_data_value
        full_vector[: len(np.ravel(values))] = np.ravel(values)

        return full_vector

    def __call__(self):
        return self.values
