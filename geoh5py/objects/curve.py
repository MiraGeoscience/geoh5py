import uuid
from typing import List, Optional, Union

from numpy import arange, asarray, c_, ndarray, unique, vstack, where, zeros

from .object_base import ObjectType
from .points import Points


class Curve(Points):
    """
    A Curve object is defined by a series of cells (segments) connecting a set of
    vertices. Data can be associated to both the cells and vertices.
    """

    __TYPE_UID = uuid.UUID(
        fields=(0x6A057FDC, 0xB355, 0x11E3, 0x95, 0xBE, 0xFD84A7FFCB88)
    )

    def __init__(self, object_type: ObjectType, **kwargs):

        self._cells: Optional[ndarray] = None
        self._parts: Optional[ndarray] = None
        super().__init__(object_type, **kwargs)

        if object_type.name == "None":
            self.entity_type.name = "Curve"

    @property
    def cells(self) -> Optional[ndarray]:
        """
        Array of indices defining the connection between vertices:
        numpy.ndarray of int, shape ("*", 2)
        """
        if getattr(self, "_cells", None) is None:
            if self._parts is not None:
                cells = []
                for part_id in self.unique_parts:
                    ind = where(self.parts == part_id)[0]
                    cells.append(c_[ind[:-1], ind[1:]])
                self._cells = vstack(cells)

            elif self.existing_h5_entity:
                self._cells = self.workspace.fetch_cells(self.uid)
            else:
                if self.vertices is not None:
                    n_segments = self.vertices.shape[0]
                    self._cells = c_[
                        arange(0, n_segments - 1), arange(1, n_segments)
                    ].astype("uint32")

        return self._cells

    @cells.setter
    def cells(self, indices):
        assert indices.dtype == "uint32", "Indices array must be of type 'uint32'"
        self.modified_attributes = "cells"
        self._cells = indices
        self._parts = None

    @property
    def parts(self):
        """
        Line identifier: numpy.ndarray of int, shape (n_vertices, )
        """
        if getattr(self, "_parts", None) is None and self.cells is not None:

            cells = self.cells
            parts = zeros(self.vertices.shape[0], dtype="int")
            count = 0
            for ind in range(1, cells.shape[0]):

                if cells[ind, 0] != cells[ind - 1, 1]:
                    count += 1

                parts[cells[ind, :]] = count

            self._parts = parts

        return self._parts

    @parts.setter
    def parts(self, indices: Union[List, ndarray]):
        if self.vertices is not None:
            if isinstance(indices, list):
                indices = asarray(indices)

            assert indices.dtype == int, "Indices array must be of type 'uint32'"
            assert indices.shape == (
                self.vertices.shape[0],
            ), f"Provided parts must be of shape {self.vertices.shape[0]}"

            self.modified_attributes = "cells"
            self._parts = indices
            self._cells = None

    @property
    def unique_parts(self):
        """
        Unique parts connected by cells
        """
        if self.parts is not None:

            return unique(self.parts).tolist()

        return None

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        """
        :return: Default unique identifier
        """
        return cls.__TYPE_UID


class SurveyAirborneMagnetics(Curve):
    """
    An airborne magnetic survey object
    """

    __TYPE_UID = uuid.UUID(
        fields=(0x4B99204C, 0xD133, 0x4579, 0xA9, 0x16, 0xA9C8B98CFCCB)
    )

    def __init__(self, object_type: ObjectType, **kwargs):
        super().__init__(object_type, **kwargs)

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        """
        :return: Default unique identifier
        """
        return cls.__TYPE_UID
