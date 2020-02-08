import uuid
from typing import Optional

from numpy import arange, c_, ndarray

from .cell import Cell
from .object_base import ObjectType
from .points import Points


class Curve(Points):
    """
    A ``Curve`` object is defined by a series of cells (segments)
    connecting a set of vertices (points). Data can be associated to both the
    cells and vertices.
    """

    __TYPE_UID = uuid.UUID(
        fields=(0x6A057FDC, 0xB355, 0x11E3, 0x95, 0xBE, 0xFD84A7FFCB88)
    )

    def __init__(self, object_type: ObjectType, **kwargs):

        self._cells: Optional[Cell] = None
        super().__init__(object_type, **kwargs)

        if object_type.name == "None":
            self.entity_type.name = "Curve"

    @property
    def cells(self) -> Optional[ndarray]:
        """
        Array of indices defining the connection between vertices:
        array of int, shape (*, 2)
        """
        if getattr(self, "_cells", None) is None:
            if self.existing_h5_entity:
                self._cells = self.workspace.fetch_cells(self.uid)
            else:
                if self.vertices is not None:
                    n_segments = self.vertices.locations.shape[0]
                    self._cells = c_[
                        arange(0, n_segments - 1), arange(1, n_segments)
                    ].astype("uint32")

        return self._cells

    @cells.setter
    def cells(self, indices):
        assert indices.dtype == "uint32", "Indices array must be of type 'uint32'"
        self.update_h5 = "cells"
        self._cells = indices

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        """
        :return: Default unique identifier
        """
        return cls.__TYPE_UID
