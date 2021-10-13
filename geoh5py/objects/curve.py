#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoh5py.
#
#  geoh5py is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  geoh5py is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with geoh5py.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import uuid

import numpy as np

from .object_base import ObjectType
from .points import Points


class Curve(Points):
    """
    Curve object defined by a series of line segments (:obj:`~geoh5py.objects.curve.Curve.cells`)
    connecting :obj:`~geoh5py.objects.object_base.ObjectBase.vertices`.
    """

    _attribute_map = Points._attribute_map.copy()
    _attribute_map.update(
        {
            "Last focus": "last_focus",
            "PropertyGroups": "property_groups",
            "Current line property ID": "current_line_id",
        }
    )

    __TYPE_UID = uuid.UUID(
        fields=(0x6A057FDC, 0xB355, 0x11E3, 0x95, 0xBE, 0xFD84A7FFCB88)
    )

    def __init__(self, object_type: ObjectType, **kwargs):

        self._cells: np.ndarray | None = None
        self._parts: np.ndarray | None = None
        super().__init__(object_type, **kwargs)

    @property
    def cells(self) -> np.ndarray | None:
        r"""
        :obj:`numpy.ndarray` of :obj:`int`, shape (\*, 2):
        Array of indices defining segments connecting vertices. Defined based on
        :obj:`~geoh5py.objects.curve.Curve.parts` if set by the user.
        """
        if getattr(self, "_cells", None) is None:
            if self._parts is not None:
                cells = []
                for part_id in self.unique_parts:
                    ind = np.where(self.parts == part_id)[0]
                    cells.append(np.c_[ind[:-1], ind[1:]])
                self._cells = np.vstack(cells)

            elif self.existing_h5_entity:
                self._cells = self.workspace.fetch_cells(self.uid)
            else:
                if self.vertices is not None:
                    n_segments = self.vertices.shape[0]
                    self._cells = np.c_[
                        np.arange(0, n_segments - 1), np.arange(1, n_segments)
                    ].astype("uint32")

        return self._cells

    @cells.setter
    def cells(self, indices):
        assert np.issubdtype(
            indices.dtype, np.integer
        ), "Indices array must be of integer type"
        self.modified_attributes = "cells"
        self._cells = indices.astype(np.int32)
        self._parts = None

    @property
    def current_line_id(self):

        if getattr(self, "_current_line_id", None) is None:
            self._current_line_id = uuid.uuid4()

        return self._current_line_id

    @current_line_id.setter
    def current_line_id(self, value: uuid.UUID):

        if isinstance(value, str):
            value = uuid.UUID(value)

        assert isinstance(value, uuid.UUID), (
            f"Input current_line_id value should be of type {uuid.UUID}."
            f" {type(value)} provided"
        )

        self._current_line_id = value
        self.modified_attributes = "attributes"

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        """
        :return: Default unique identifier
        """
        return cls.__TYPE_UID

    @property
    def parts(self):
        """
        :obj:`numpy.array` of :obj:`int`, shape
        (:obj:`~geoh5py.objects.object_base.ObjectBase.n_vertices`, 2):
        Group identifiers for vertices connected by line segments as defined by the
        :obj:`~geoh5py.objects.curve.Curve.cells`
        property. The definition of the :obj:`~geoh5py.objects.curve.Curve.cells`
        property get modified by the setting of parts.
        """
        if getattr(self, "_parts", None) is None and self.cells is not None:

            cells = self.cells
            parts = np.zeros(self.vertices.shape[0], dtype="int")
            count = 0
            for ind in range(1, cells.shape[0]):

                if cells[ind, 0] != cells[ind - 1, 1]:
                    count += 1

                parts[cells[ind, :]] = count

            self._parts = parts

        return self._parts

    @parts.setter
    def parts(self, indices: list | np.ndarray):
        if self.vertices is not None:
            if isinstance(indices, list):
                indices = np.asarray(indices, dtype="int32")
            else:
                indices = indices.astype("int32")

            assert indices.shape == (
                self.vertices.shape[0],
            ), f"Provided parts must be of shape {self.vertices.shape[0]}"

            self.modified_attributes = "cells"
            self._parts = indices
            self._cells = None

    @property
    def unique_parts(self):
        """
        :obj:`list` of :obj:`int`: Unique :obj:`~geoh5py.objects.curve.Curve.parts`
        identifiers.
        """
        if self.parts is not None:

            return np.unique(self.parts).tolist()

        return None
