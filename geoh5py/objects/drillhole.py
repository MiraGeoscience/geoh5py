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
from typing import List, Optional, Text, Union

import numpy as np

from ..data.data import Data
from .object_base import ObjectType, validate_data_type
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
        self._locations = None
        self._deviation_x = None
        self._deviation_y = None
        self._deviation_z = None
        self._deviation_length = None

        super().__init__(object_type, **kwargs)

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID

    def add_data(
        self, data: dict, property_group: str = None
    ) -> Union[Data, List[Data]]:
        """
        Create :obj:`~geoh5py.data.data.Data` specific to the drillhole object
        from dictionary of name and arguments. A keyword 'depth' or 'from-to'
        with corresponding depth values is expected in order to locate the
        data along the well path.

        :param data: Dictionary of data to be added to the object, e.g.

        .. code-block:: python

            data_dict = {
                "data_A": {
                    'values', [v_1, v_2, ...],
                    "from-to": numpy.ndarray,
                    },
                "data_B": {
                    'values', [v_1, v_2, ...],
                    "depth": numpy.ndarray,
                    },
            }

        :return: List of new Data objects.
        """
        data_objects = []

        for name, attr in data.items():
            assert isinstance(attr, dict), (
                f"Given value to data {name} should of type {dict}. "
                f"Type {type(attr)} given instead."
            )
            assert "values" in list(
                attr.keys()
            ), f"Given attr for data {name} should include 'values'"

            attr["name"] = name

            if "depth" in list(attr.keys()):
                attr["association"] = "VERTEX"
                attr["values"] = self.validate_log_data(attr["depth"], attr["values"])
            elif "from-to" in list(attr.keys()):
                attr["association"] = "CELL"
                attr["values"] = self.validate_log_data(attr["depth"], attr["values"])
            else:
                assert attr["association"] == "OBJECT", (
                    "Input data dictionary must contain {key:values} "
                    + "{'depth':numpy.ndarray}, {'from-to':numpy.ndarray} "
                    + "or {'association': 'OBJECT'}."
                )

            entity_type = validate_data_type(attr)
            kwargs = {"parent": self, "association": attr["association"]}
            for key, val in attr.items():
                if key in ["parent", "association", "entity_type", "type"]:
                    continue
                kwargs[key] = val

            data_object = self.workspace.create_entity(
                Data, entity=kwargs, entity_type=entity_type
            )

            if property_group is not None:
                self.add_data_to_group(data_object, property_group)

            data_objects.append(data_object)

        if len(data_objects) == 1:
            return data_object

        return data_objects

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
    def deviation_length(self):
        """
        Store the survey lengths
        """
        if (
            getattr(self, "_deviation_length", None) is None
            and self.surveys is not None
        ):
            self._deviation_length = (
                self.surveys["Depth"][1:] - self.surveys["Depth"][:-1]
            )

        return self._deviation_length

    @property
    def deviation_x(self):
        """
        :obj:`numpy.ndarray`: Store the change in x-coordinates along the well path.
        """
        if getattr(self, "_deviation_x", None) is None and self.surveys is not None:
            dx_in = np.cos(
                np.deg2rad(450.0 - self.surveys["Azimuth"][:-1] % 360.0)
            ) * np.cos(np.deg2rad(self.surveys["Dip"][:-1]))
            dx_out = np.cos(
                np.deg2rad(450.0 - self.surveys["Azimuth"][1:] % 360.0)
            ) * np.cos(np.deg2rad(self.surveys["Dip"][1:]))
            ddx = (dx_out - dx_in) / self.deviation_length / 2.0
            self._deviation_x = dx_in + self.deviation_length * ddx

        return self._deviation_x

    @property
    def deviation_y(self):
        """
        :obj:`numpy.ndarray`: Store the change in y-coordinates along the well path.
        """
        if getattr(self, "_deviation_y", None) is None and self.surveys is not None:
            dy_in = np.sin(
                np.deg2rad(450.0 - self.surveys["Azimuth"][:-1] % 360.0)
            ) * np.cos(np.deg2rad(self.surveys["Dip"][:-1]))
            dy_out = np.sin(
                np.deg2rad(450.0 - self.surveys["Azimuth"][1:] % 360.0)
            ) * np.cos(np.deg2rad(self.surveys["Dip"][1:]))
            ddy = (dy_out - dy_in) / self.deviation_length / 2.0
            self._deviation_y = dy_in + self.deviation_length * ddy

        return self._deviation_y

    @property
    def deviation_z(self):
        """
        :obj:`numpy.ndarray`: Store the change in z-coordinates along the well path.
        """
        if getattr(self, "_deviation_z", None) is None and self.surveys is not None:
            dz_in = np.sin(np.deg2rad(self.surveys["Dip"][:-1]))
            dz_out = np.sin(np.deg2rad(self.surveys["Dip"][1:]))
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

        indices = np.searchsorted(self.surveys["Depth"], depths, side="left") - 1

        locations = (
            self.locations[indices, :]
            + (depths - self.surveys["Depth"][indices])[:, None]
            * np.c_[
                self.deviation_x[indices],
                self.deviation_y[indices],
                self.deviation_z[indices],
            ]
        )

        return locations

    def validate_log_data(self, depth, input_values, threshold=1e-4):
        """
        Compare new and current depth values, append new vertices if necessary and return
        an augmented values vector that matches the vertices indexing.
        """

        values = input_values
        if "DEPTH" not in self.get_data_list():  # First data appended
            self.vertices = self.desurvey(depth)
            self.workspace.create_entity(
                Data,
                entity={
                    "parent": self,
                    "association": "VERTEX",
                    "name": "DEPTH",
                    "values": depth,
                },
                entity_type={"primitive_type": "FLOAT"},
            )
        else:
            depth_obj = self.get_data("DEPTH")[0]
            r_nn = np.searchsorted(depth_obj.values, depth, side="right")
            lr_nn = np.c_[r_nn, r_nn - 1]
            match = np.where(
                np.abs(depth_obj.values[lr_nn] - depth[:, None]) < threshold
            )
            indices = np.c_[match[0], lr_nn[match[0], match[1]]]

            if np.any(indices):
                values = np.zeros(self.n_vertices)
                values[indices[:, 1]] = input_values[indices[:, 0]]
                values = np.r_[values, np.delete(input_values, indices[:, 0])]
                depth_obj.values = np.r_[
                    depth_obj.values, np.delete(depth, indices[:, 0])
                ]
                self.vertices = self.desurvey(depth_obj.values)

        return values

    # def validate_data_association(self, attribute_dict):
    #     """
    #     Get a dictionary of attributes and validate the data 'association'
    #     with special actions for drillhole objects.
    #     """
    #
    #     if "depth" in list(attribute_dict.keys()):
    #         attribute_dict["association"] = "VERTEX"
    #
    #     elif "from_to" in list(attribute_dict.keys()):
    #         attribute_dict["association"] = "CELL"
    #
    #     if "association" in list(attribute_dict.keys()):
    #         assert attribute_dict["association"] in [
    #             enum.name for enum in DataAssociationEnum
    #         ], (
    #             "Data 'association' must be one of "
    #             + f"{[enum.name for enum in DataAssociationEnum]}. "
    #             + f"{attribute_dict['association']} provided."
    #         )
