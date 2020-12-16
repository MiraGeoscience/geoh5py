#  Copyright (c) 2020 Mira Geoscience Ltd.
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

import uuid
from typing import Optional, Text

import numpy as np

from .object_base import ObjectType
from .points import Points


class Drillhole(Points):
    """
    Drillhole object class defined by

    .. warning:: Not yet implemented.

    """

    __TYPE_UID = uuid.UUID(
        fields=(0x7CAEBF0E, 0xD16E, 0x11E3, 0xBC, 0x69, 0xE4632694AA37)
    )

    _attribute_map = Points._attribute_map.copy()
    _attribute_map.update({"Cost": "cost", "Collar": "collar", "Planning": "planning"})

    def __init__(self, object_type: ObjectType, **kwargs):

        # TODO
        self._vertices: Optional[np.ndarray] = None
        self._cells: Optional[np.ndarray] = None
        self._collar: Optional[np.ndarray] = None
        self._surveys: Optional[np.ndarray] = None
        self._trace: Optional[np.ndarray] = None
        self._trace_depth: Optional[np.ndarray] = None
        self._cost: Optional[float] = 0.0
        self._planning: Text = "Default"

        super().__init__(object_type, **kwargs)

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID

    @property
    def cells(self) -> Optional[np.ndarray]:
        r"""
        :obj:`numpy.ndarray` of :obj:`int`, shape (\*, 2):
        Array of indices defining segments connecting vertices.
        """
        if getattr(self, "_cells", None) is None:
            if self.existing_h5_entity:
                self._cells = self.workspace.fetch_cells(self.uid)

        return self._cells

    @cells.setter
    def cells(self, indices):
        assert indices.dtype == "uint32", "Indices array must be of type 'uint32'"
        self.modified_attributes = "cells"
        self._cells = indices

    @property
    def collar(self):
        """
        :obj:`numpy.array` of :obj:`float`, shape (3, ): Coordinates of the collar
        """
        return self._collar

    @collar.setter
    def collar(self, value):
        if value is not None:
            if isinstance(value, np.ndarray):
                value = value.tolist()

            assert len(value) == 3, "Origin must be a list or numpy array of shape (3,)"

            self.modified_attributes = "attributes"
            self._centroids = None

            value = np.asarray(
                tuple(value), dtype=[("x", float), ("y", float), ("z", float)]
            )
            self._collar = value

    @property
    def cost(self):
        """
        Cost estimate of the drillhole
        """
        return self._cost

    @cost.setter
    def cost(self, value):
        assert isinstance(value, float), f"Provided cost value must be of type {float}"
        self._cost = value

    @property
    def surveys(self):
        """
        :obj:`numpy.array` of :obj:`float`, shape (3, ): Coordinates of the surveys
        """
        if (getattr(self, "_surveys", None) is None) and self.existing_h5_entity:
            surveys = self.workspace.fetch_coordinates(self.uid, "surveys")
            self._surveys = np.c_[surveys["Depth"], surveys["Dip"], surveys["Azimuth"]]

        return self._surveys

    @surveys.setter
    def surveys(self, value):
        if value is not None:
            value = np.vstack(value)

            assert value.shape[1] == 3, "'surveys' requires an ndarray of shape (*, 3)"
            self.modified_attributes = "surveys"
            self._surveys = np.core.records.fromarrays(
                value.T, names="Depth, Dip, Azimuth", formats="<f8, <f8, <f8, <f8"
            )

    @property
    def trace(self) -> Optional[np.ndarray]:
        """
        Drillhole trace defining the path in 3D
        """
        if (getattr(self, "_trace", None) is None) and self.existing_h5_entity:
            trace = self.workspace.fetch_coordinates(self.uid, "trace")
            self._trace = np.c_[trace["x"], trace["y"], trace["z"]]

        return self._trace

    @trace.setter
    def trace(self, xyz: np.ndarray):
        self.modified_attributes = "trace"
        self._trace = xyz

    @property
    def trace_depth(self) -> Optional[np.ndarray]:
        """
        Drillhole trace depth from top to bottom
        """
        if getattr(self, "_trace_depth", None) is None and self.trace is not None:
            trace = self.trace
            self._trace_depth = trace[0, 2] - trace[:, 2]

        return self._trace_depth

    @property
    def planning(self):
        """
        Cost estimate of the drillhole
        """
        return self._planning

    @planning.setter
    def planning(self, value):
        choices = ["Default", "Ongoing", "Planned", "Completed"]
        assert value in choices, f"Provided planning value must be one of {choices}"
        self._planning = value
