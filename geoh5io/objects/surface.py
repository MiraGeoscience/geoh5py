import uuid
from typing import Optional

from numpy import arange, c_, ndarray

from .cell import Cell
from .object_base import ObjectType
from .points import Points


class Surface(Points):
    __TYPE_UID = uuid.UUID(
        fields=(0xF26FEBA3, 0xADED, 0x494B, 0xB9, 0xE9, 0xB2BBCBE298E1)
    )

    def __init__(self, object_type: ObjectType, **kwargs):

        self._cells: Optional[Cell] = None

        super().__init__(object_type, **kwargs)

        if object_type.name == "None":
            self.entity_type.name = "Surface"

    @property
    def cells(self) -> Optional[ndarray]:
        """
        @property
        cells(xyz)

        Function to return the object cells coordinates.

        Returns
        -------
        cells: geoh5io.objects.Cell
            Cell object holding vertices index
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
    def cells(self, indices: ndarray):
        """
        @property.setter

        cells(id1, id2, id3)

        Parameters
        ----------
        indices: numpy.array
            Integer values [n x 3]

        Returns
        -------
        cells: geoh5io.objects.Cell
            Cell object holding vertices index
        """
        assert indices.dtype in [
            "int32",
            "uint32",
        ], "Indices array must be of type 'uint32'"

        if indices.dtype == "int32":
            indices.astype("uint32")
        self.modified_entity = "cells"
        self._cells = indices

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID

    @property
    def n_cells(self):
        """
        n_cells

        Returns
        -------
        n_cells: int
            Number of cells
        """
        if self.cells is not None:
            return self._cells.shape[0]
        return None
