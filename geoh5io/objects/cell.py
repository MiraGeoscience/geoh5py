from numpy import ndarray


class Cell:
    def __init__(self, indices: ndarray):

        assert indices.dtype == "int32", "Indices array must be of type 'int32'"
        self._indices = indices

    @property
    def indices(self) -> ndarray:
        return self._indices

    def __call__(self):
        return self._indices
