import uuid
from typing import Optional

from numpy import arange, c_, ndarray

from .cell import Cell
from .object_base import ObjectType
from .points import Points


class Curve(Points):
    __TYPE_UID = uuid.UUID(
        fields=(0x6A057FDC, 0xB355, 0x11E3, 0x95, 0xBE, 0xFD84A7FFCB88)
    )

    attribute_map = {
        "Allow delete": "allow_delete",
        "Allow move": "allow_rename",
        "Allow rename": "allow_rename",
        "ID": "uid",
        "Last focus": "last_focus",
        "Name": "name",
        "Public": "public",
        "PropertyGroups": "property_groups",
    }

    def __init__(self, object_type: ObjectType, name: str, uid: uuid.UUID = None):
        super().__init__(object_type, name, uid)

        self._cells: Optional[Cell] = None

        if object_type.name is None:
            self.entity_type.name = "Curve"
        else:
            self.entity_type.name = object_type.name

        if object_type.description is None:
            self.entity_type.description = "Curve"
        else:
            self.entity_type.description = object_type.description

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
    def cells(self, indices):
        """
        @property.setter

        cells(id1, id2)

        Parameters
        ----------
        indices: numpy.array
            Integer values [n x 2]

        Returns
        -------
        cells: geoh5io.objects.Cell
            Cell object holding vertices index
        """
        assert indices.dtype == "uint32", "Indices array must be of type 'uint32'"

        self._cells = indices

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID
