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

import uuid
from typing import Optional, Text

import numpy as np

from ..data.data_association_enum import DataAssociationEnum
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
    _attribute_map.update(
        {
            "Cost": "cost",
            "Collar": "collar",
            "Planning": "planning",
        }
    )

    def __init__(self, object_type: ObjectType, **kwargs):

        self._cells: Optional[np.ndarray] = None
        self._collar: Optional[np.ndarray] = None
        self._cost: Optional[float] = 0.0
        self._planning: Text = "Default"
        self._surveys: np.recarray = None
        self._trace: np.recarray = None
        self._trace_depth: Optional[np.ndarray] = None
        self._depth = None
        self._locations = None
        self._deviation_x = None
        self._deviation_y = None
        self._deviation_z = None
        self._deviation_length = None

        super().__init__(object_type, **kwargs)

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID

    @property
    def azimuth(self):
        """
        :obj:`numpy.ndarray`: Pointer to
        :obj:`~geoh5py.objects.drillole.Drillhole.surveys` 'Azimuth' values.
        """
        if self.surveys is not None:
            return self.surveys["Azimuth"]

        return None

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
        :obj:`float`: Cost estimate of the drillhole
        """
        return self._cost

    @cost.setter
    def cost(self, value):
        assert isinstance(value, float), f"Provided cost value must be of type {float}"
        self._cost = value

    @property
    def depth(self):
        """
        :obj:`numpy.ndarray`: Pointer to
        :obj:`~geoh5py.objects.drillole.Drillhole.surveys` 'Depth' values.
        """
        if self.surveys is not None:
            return self.surveys["Depth"]

        return None

    @property
    def dip(self):
        """
        :obj:`numpy.ndarray`: Pointer to
        :obj:`~geoh5py.objects.drillole.Drillhole.surveys` 'Dip' values.
        """
        if self.surveys is not None:
            return self.surveys["Dip"]

        return None

    @property
    def deviation_length(self):
        """
        Store the survey lengths
        """
        if (
            getattr(self, "_deviation_length", None) is None
            and self.surveys is not None
        ):
            self._deviation_length = self.depth[1:] - self.depth[:-1]

        return self._deviation_length

    @property
    def deviation_x(self):
        """
        :obj:`numpy.ndarray`: Store the change in x-coordinates along the well path.
        """
        if getattr(self, "_deviation_x", None) is None and self.surveys is not None:
            dx_in = np.cos(np.deg2rad(450.0 - self.azimuth[:-1] % 360.0)) * np.cos(
                np.deg2rad(self.dip[:-1])
            )
            dx_out = np.cos(np.deg2rad(450.0 - self.azimuth[1:] % 360.0)) * np.cos(
                np.deg2rad(self.dip[1:])
            )
            ddx = (dx_out - dx_in) / self.deviation_length / 2.0
            self._deviation_x = dx_in + self.deviation_length * ddx

        return self._deviation_x

    @property
    def deviation_y(self):
        """
        :obj:`numpy.ndarray`: Store the change in y-coordinates along the well path.
        """
        if getattr(self, "_deviation_y", None) is None and self.surveys is not None:
            dy_in = np.sin(np.deg2rad(450.0 - self.azimuth[:-1] % 360.0)) * np.cos(
                np.deg2rad(self.dip[:-1])
            )
            dy_out = np.sin(np.deg2rad(450.0 - self.azimuth[1:] % 360.0)) * np.cos(
                np.deg2rad(self.dip[1:])
            )
            ddy = (dy_out - dy_in) / self.deviation_length / 2.0
            self._deviation_y = dy_in + self.deviation_length * ddy

        return self._deviation_y

    @property
    def deviation_z(self):
        """
        :obj:`numpy.ndarray`: Store the change in z-coordinates along the well path.
        """
        if getattr(self, "_deviation_z", None) is None and self.surveys is not None:
            dz_in = np.sin(np.deg2rad(self.dip[:-1]))
            dz_out = np.sin(np.deg2rad(self.dip[1:]))
            ddz = (dz_out - dz_in) / self.deviation_length / 2.0
            self._deviation_z = dz_in + self.deviation_length * ddz

        return self._deviation_z

    @property
    def locations(self):
        """
        :obj:`numpy.ndarray`: Lookup array of the well path x,y,z coordinates.
        """
        if (
            getattr(self, "_locations", None) is None
            and self.collar is not None
            and self.surveys is not None
        ):
            self._locations = np.c_[
                self.collar["x"] + np.cumsum(self.deviation_length * self.deviation_x),
                self.collar["y"] + np.cumsum(self.deviation_length * self.deviation_y),
                self.collar["z"] + np.cumsum(self.deviation_length * self.deviation_z),
            ]

        return self._locations

    @property
    def surveys(self):
        """
        :obj:`numpy.array` of :obj:`float`, shape (3, ): Coordinates of the surveys
        """
        if (getattr(self, "_surveys", None) is None) and self.existing_h5_entity:
            self._surveys = self.workspace.fetch_coordinates(self.uid, "surveys")

        if getattr(self, "_surveys", None) is not None:
            return self._surveys

        return None

    @surveys.setter
    def surveys(self, value):
        if value is not None:
            value = np.vstack(value)

            assert value.shape[1] == 3, "'surveys' requires an ndarray of shape (*, 3)"
            self.modified_attributes = "surveys"
            self._surveys = np.core.records.fromarrays(
                value.T, names="Depth, Dip, Azimuth", formats="<f4, <f4, <f4"
            )

            # Reset the trace
            self.modified_attributes = "trace"
            self._trace = None

    @property
    def trace(self) -> Optional[np.ndarray]:
        """
        :obj:`numpy.array`: Drillhole trace defining the path in 3D
        """
        if (getattr(self, "_trace", None) is None) and self.existing_h5_entity:
            self._trace = self.workspace.fetch_coordinates(self.uid, "trace")

        if getattr(self, "_trace", None) is not None:
            return self._trace.view("<f8").reshape((-1, 3))

        return None

    @property
    def trace_depth(self) -> Optional[np.ndarray]:
        """
        :obj:`numpy.array`: Drillhole trace depth from top to bottom
        """
        if getattr(self, "_trace_depth", None) is None and self.trace is not None:
            trace = self.trace
            self._trace_depth = trace[0, 2] - trace[:, 2]

        return self._trace_depth

    @property
    def planning(self):
        """
        :obj:`str`: Status of the hole: ["Default", "Ongoing", "Planned", "Completed"]
        """
        return self._planning

    @planning.setter
    def planning(self, value):
        choices = ["Default", "Ongoing", "Planned", "Completed"]
        assert value in choices, f"Provided planning value must be one of {choices}"
        self._planning = value

    def add_depth_vertices(self, depth):
        """
        Get a list of depths to be converted to vertices along the well path
        """

    def desurvey(self, depths):
        """
        Function to return x, y, z coordinates from depth.
        """
        assert (
            self.surveys is not None and self.collar is not None
        ), "'surveys' and 'collar' attributes required for desurvey operation"

        if isinstance(depths, list):
            depths = np.asarray(depths)

        indices = np.searchsorted(self.depth, depths, side="left") - 1

        locations = (
            self.locations[indices, :]
            + (depths - self.depth[indices])[:, None]
            * np.c_[
                self.deviation_x[indices],
                self.deviation_y[indices],
                self.deviation_z[indices],
            ]
        )

        return locations

    def validate_data_association(self, attribute_dict):
        """
        Get a dictionary of attributes and validate the data 'association'
        with special actions for drillhole objects.
        """

        if "depth" in list(attribute_dict.keys()):
            attribute_dict["association"] = "VERTEX"

        elif "from_to" in list(attribute_dict.keys()):
            attribute_dict["association"] = "CELL"

        if "association" in list(attribute_dict.keys()):
            assert attribute_dict["association"] in [
                enum.name for enum in DataAssociationEnum
            ], (
                "Data 'association' must be one of "
                + f"{[enum.name for enum in DataAssociationEnum]}. "
                + f"{attribute_dict['association']} provided."
            )
