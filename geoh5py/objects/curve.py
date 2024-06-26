#  Copyright (c) 2024 Mira Geoscience Ltd.
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

from ..shared.utils import str2uuid
from .cell_object import CellObject
from .object_base import ObjectType


class Curve(CellObject):
    """
    Curve object defined by a series of line segments (:obj:`~geoh5py.objects.curve.Curve.cells`)
    connecting :obj:`~geoh5py.objects.object_base.ObjectBase.vertices`.
    """

    _attribute_map: dict = CellObject._attribute_map.copy()
    _attribute_map.update(
        {
            "Current line property ID": "current_line_id",
        }
    )

    __TYPE_UID = uuid.UUID(
        fields=(0x6A057FDC, 0xB355, 0x11E3, 0x95, 0xBE, 0xFD84A7FFCB88)
    )

    def __init__(  # pylint: disable="too-many-arguments"
        self,
        object_type: ObjectType,
        vertices: np.ndarray = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        cells: np.ndarray | None = None,
        current_line_id: uuid.UUID | None = None,
        parts: np.ndarray | None = None,
        name="Curve",
        **kwargs,
    ):
        self._current_line_id: uuid.UUID | None = None
        self._parts: np.ndarray | None
        self._vertices: np.ndarray = self.validate_vertices(vertices)

        if parts is not None:
            if cells is not None:
                raise ValueError("Parts can only be set if cells are not provided.")

            self._parts = self.validate_parts(parts)
            cells = self.make_cells_from_parts()

        super().__init__(
            object_type,
            cells=cells,
            current_line_id=current_line_id,
            name=name,
            vertices=vertices,
            **kwargs,
        )

    @property
    def cells(self) -> np.ndarray:
        r"""
        :obj:`numpy.ndarray` of :obj:`int`, shape (\*, 2):
        Array of indices defining segments connecting vertices. Defined based on
        :obj:`~geoh5py.objects.curve.Curve.parts` if set by the user.
        """
        if self._cells is None and self.on_file:
            self._cells = self.workspace.fetch_array_attribute(self, "cells")

        return self._cells

    @property
    def current_line_id(self) -> uuid.UUID | None:
        """
        :obj:`uuid.UUID` or :obj:`None`: Unique identifier of the current line.
        """
        return self._current_line_id

    @current_line_id.setter
    def current_line_id(self, value: uuid.UUID | None):
        value = str2uuid(value)

        if not isinstance(value, (uuid.UUID, type(None))):
            raise TypeError(
                f"Input current_line_id value should be of type {uuid.UUID}."
                f" {type(value)} provided"
            )

        self._current_line_id = value

        if self.on_file:
            self.workspace.update_attribute(self, "attributes")

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        """
        :return: Default unique identifier
        """
        return cls.__TYPE_UID

    def make_cells_from_parts(self) -> np.ndarray | None:
        """
        Generate cells from parts.
        """
        if self.unique_parts is None:
            return None

        cells = []
        for part_id in self.unique_parts:
            ind = np.where(self.parts == part_id)[0]
            cells.append(np.sort(np.c_[ind[:-1], ind[1:]], axis=0))
        return np.vstack(cells)

    @property
    def parts(self) -> np.ndarray:
        """
        :obj:`numpy.array` of :obj:`int`, shape
        (:obj:`~geoh5py.objects.object_base.ObjectBase.n_vertices`, 2):
        Group identifiers for vertices connected by line segments as defined by the
        :obj:`~geoh5py.objects.curve.Curve.cells`
        property. The definition of the :obj:`~geoh5py.objects.curve.Curve.cells`
        property get modified by the setting of parts.
        """
        if getattr(self, "_parts", None) is None:
            cells = self.cells
            parts = np.zeros(self.vertices.shape[0], dtype="int")
            count = 0
            for ind in range(1, cells.shape[0]):
                if cells[ind, 0] != cells[ind - 1, 1]:
                    count += 1

                parts[cells[ind, :]] = count

            self._parts = parts

        return self._parts

    @property
    def unique_parts(self):
        """
        :obj:`list` of :obj:`int`: Unique :obj:`~geoh5py.objects.curve.Curve.parts`
        identifiers.
        """
        return np.unique(self.parts).tolist()

    def validate_cells(self, indices: tuple | list | np.ndarray | None):
        """
        Validate or generate cells array.

        :param indices: Array of indices defining segments connecting vertices.
        """
        if indices is None:
            n_segments = self.vertices.shape[0]
            indices = np.c_[
                np.arange(0, n_segments - 1), np.arange(1, n_segments)
            ].astype("uint32")

        if isinstance(indices, (list, tuple)):
            indices = np.array(indices, ndmin=2)

        if indices.ndim != 2 or indices.shape[1] != 2:
            raise ValueError("Array of cells should be of shape (*, 2).")

        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("Indices array must be of integer type")

        return indices

    def validate_parts(self, indices: list | tuple | np.ndarray):
        """
        Validate parts array.

        :param indices: Array of indices defining group of vertices belonging to
            connected cell segments.
        """
        if isinstance(indices, (list | tuple)):
            indices = np.asarray(indices, dtype="int32")

        if not isinstance(indices, np.ndarray):
            raise TypeError("Parts must be a list or numpy array.")

        indices = indices.astype("int32")

        if len(indices) != self.n_vertices:
            raise ValueError(
                f"Provided parts must be of shape {self.vertices.shape[0]}"
            )

        return indices

    @classmethod
    def validate_vertices(cls, xyz: np.ndarray | list | tuple) -> np.ndarray:
        """
        Validate and format type of vertices array.

        :param xyz: Array of vertices as defined by :obj:`~geoh5py.objects.points.Points.vertices`.
        """
        xyz = super().validate_vertices(xyz)

        if len(xyz) < 2:
            raise ValueError("Curve must have at least two vertices.")

        return xyz
