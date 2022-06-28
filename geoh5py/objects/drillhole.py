#  Copyright (c) 2022 Mira Geoscience Ltd.
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

# pylint: disable=R0902

from __future__ import annotations

import re
import uuid

import numpy as np

from ..data import Data, FloatData
from ..shared.utils import match_values, merge_arrays
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
            "End of hole": "end_of_hole",
        }
    )

    def __init__(self, object_type: ObjectType, **kwargs):
        self._cells: np.ndarray | None = None
        self._collar: np.ndarray | None = None
        self._cost: float | None = 0.0
        self._depths: FloatData | None = None
        self._end_of_hole: float | None = None
        self._planning: str = "Default"
        self._surveys: np.ndarray | None = None
        self._trace: np.ndarray | None = None
        self._trace_depth: np.ndarray | None = None
        self._locations = None
        self._default_collocation_distance = 1e-2

        super().__init__(object_type, **kwargs)

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID

    @property
    def cells(self) -> np.ndarray | None:
        r"""
        :obj:`numpy.ndarray` of :obj:`int`, shape (\*, 2):
        Array of indices defining segments connecting vertices.
        """
        if getattr(self, "_cells", None) is None:
            if self.on_file:
                self._cells = self.workspace.fetch_array_attribute(self)

        return self._cells

    @cells.setter
    def cells(self, indices):
        assert indices.dtype == "uint32", "Indices array must be of type 'uint32'"
        self._cells = indices
        self.workspace.update_attribute(self, "cells")

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

            if isinstance(value, str):
                value = [float(n) for n in re.findall(r"\d+\.\d+", value)]

            assert len(value) == 3, "Origin must be a list or numpy array of shape (3,)"

            value = np.asarray(
                tuple(value), dtype=[("x", float), ("y", float), ("z", float)]
            )
            self._collar = value
            self.workspace.update_attribute(self, "attributes")

        self._locations = None

        if self.trace is not None:
            self._trace = None
            self._trace_depth = None
            self.workspace.update_attribute(self, "trace")
            self.workspace.update_attribute(self, "trace_depth")

    @property
    def cost(self):
        """
        :obj:`float`: Cost estimate of the drillhole
        """
        return self._cost

    @cost.setter
    def cost(self, value: float | int):
        assert isinstance(
            value, (float, int)
        ), f"Provided cost value must be of type {float} or int."
        self._cost = value
        self.workspace.update_attribute(self, "attributes")

    @property
    def end_of_hole(self) -> float | None:
        """
        End of drillhole in meters
        """
        return self._end_of_hole

    @end_of_hole.setter
    def end_of_hole(self, value: float | int | None):
        assert isinstance(
            value, (int, float, type(None))
        ), f"Provided end_of_hole value must be of type {int}"
        self._end_of_hole = value
        self.workspace.update_attribute(self, "attributes")

    @property
    def locations(self) -> np.ndarray | None:
        """
        Lookup array of the well path in x, y, z coordinates.
        """
        if (
            getattr(self, "_locations", None) is None
            and self.collar is not None
            and self.surveys is not None
        ):
            lengths = self.surveys[1:, 0] - self.surveys[:-1, 0]
            deviation_x = compute_deviation(self.surveys, "x")
            deviation_y = compute_deviation(self.surveys, "y")
            deviation_z = compute_deviation(self.surveys, "z")
            self._locations = np.c_[
                self.collar["x"] + np.cumsum(np.r_[0.0, lengths * deviation_x]),
                self.collar["y"] + np.cumsum(np.r_[0.0, lengths * deviation_y]),
                self.collar["z"] + np.cumsum(np.r_[0.0, lengths * deviation_z]),
            ]

        return self._locations

    @property
    def planning(self) -> str:
        """
        Status of the hole on of "Default", "Ongoing", "Planned", "Completed" or "No status"
        """
        return self._planning

    @planning.setter
    def planning(self, value: str):
        choices = ["Default", "Ongoing", "Planned", "Completed", "No status"]
        assert value in choices, f"Provided planning value must be one of {choices}"
        self._planning = value
        self.workspace.update_attribute(self, "attributes")

    @property
    def surveys(self) -> np.ndarray | None:
        """
        Coordinates of the surveys
        """
        if (getattr(self, "_surveys", None) is None) and self.on_file:
            self._surveys = self.workspace.fetch_array_attribute(self, "surveys")

        if isinstance(self._surveys, np.ndarray):
            try:
                surveys = self._surveys.view("<f4").reshape((-1, 3))
            except TypeError:
                surveys = np.vstack(
                    [
                        self._surveys["Depth"],
                        self._surveys["Azimuth"],
                        self._surveys["Dip"],
                    ]
                ).T

            # Repeat first survey point at surface for de-survey interpolation
            surveys = np.vstack([surveys[0, :], surveys])
            surveys[0, 0] = 0.0

            return surveys.astype(float)

        return None

    @surveys.setter
    def surveys(self, value):
        if value is not None:
            value = np.vstack(value)

            if value.shape[1] != 3:
                raise ValueError("'surveys' requires an ndarray of shape (*, 3)")

            self._surveys = np.asarray(
                np.core.records.fromarrays(
                    value.T, names="Depth, Azimuth, Dip", formats="<f4, <f4, <f4"
                )
            )
            self.workspace.update_attribute(self, "surveys")
            self.end_of_hole = float(self._surveys["Depth"][-1])
            self._trace = None
            self.workspace.update_attribute(self, "trace")

        self._locations = None

    @property
    def default_collocation_distance(self):
        """
        Minimum collocation distance for matching depth on merge
        """
        return self._default_collocation_distance

    @default_collocation_distance.setter
    def default_collocation_distance(self, tol):
        assert tol > 0, "Tolerance value should be >0"
        self._default_collocation_distance = tol
        self.workspace.update_attribute(self, "attributes")

    @property
    def trace(self) -> np.ndarray | None:
        """
        :obj:`numpy.array`: Drillhole trace defining the path in 3D
        """
        if self._trace is None and self.on_file:
            self._trace = self.workspace.fetch_array_attribute(self, "trace")

        if self._trace is not None:
            return self._trace.view("<f8").reshape((-1, 3))

        return None

    @property
    def trace_depth(self) -> np.ndarray | None:
        """
        :obj:`numpy.array`: Drillhole trace depth from top to bottom
        """
        if getattr(self, "_trace_depth", None) is None and self.trace is not None:
            trace = self.trace
            self._trace_depth = trace[0, 2] - trace[:, 2]

        return self._trace_depth

    @property
    def _from(self):
        if self.workspace.version >= 2.0:
            obj_list = []
            for name in self.parent.index:
                if "FROM" in name:
                    obj_list += self.workspace.get_entity(
                        uuid.UUID(self.parent.index[name][0][3].decode())
                    )
            return obj_list

        data_obj = self.get_data("FROM")
        if data_obj:
            return data_obj[0]

        return None

    @property
    def _to(self):
        if self.workspace.version >= 2.0:
            obj_list = []
            for name in self.parent.index:
                if "TO" in name:
                    obj_list += self.workspace.get_entity(
                        uuid.UUID(self.parent.index[name][0][3].decode())
                    )
            return obj_list

        data_obj = self.get_data("TO")
        if data_obj:
            return data_obj[0]

        return None

    @property
    def depths(self) -> FloatData | None:
        if self._depths is None:
            data_obj = self.get_data("DEPTH")
            if data_obj and isinstance(data_obj[0], FloatData):
                self.depths = data_obj[0]
        return self._depths

    @depths.setter
    def depths(self, value: FloatData | np.ndarray | None):
        if isinstance(value, np.ndarray):
            value = self.workspace.create_entity(
                Data,
                entity={
                    "parent": self,
                    "association": "VERTEX",
                    "name": "DEPTH",
                    "values": value,
                },
                entity_type={"primitive_type": "FLOAT"},
            )

        if isinstance(value, (FloatData, type(None))):
            self._depths = value
        else:
            raise ValueError(
                f"Input '_depth' property must be of type{FloatData} or None"
            )

    def add_data(self, data: dict, property_group: str = None) -> Data | list[Data]:
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

        for name, attributes in data.items():
            assert isinstance(attributes, dict), (
                f"Given value to data {name} should of type {dict}. "
                f"Type {type(attributes)} given instead."
            )
            assert (
                "values" in attributes
            ), f"Given attributes for data {name} should include 'values'"

            attributes["name"] = name

            if attributes["name"] in self.get_data_list():
                raise UserWarning(
                    f"Data with name '{attributes['name']}' already present "
                    f"on the drillhole '{self.name}'. "
                    "Consider changing the values or renaming."
                )

            attributes, new_property_group = self.validate_data(
                attributes, property_group
            )
            entity_type = self.validate_data_type(attributes)
            kwargs = {
                "name": None,
                "parent": self,
                "association": attributes["association"],
                "allow_move": False,
            }
            for key, val in attributes.items():
                if key in ["parent", "association", "entity_type", "type"]:
                    continue
                kwargs[key] = val

            data_object = self.workspace.create_entity(
                Data, entity=kwargs, entity_type=entity_type
            )

            if not isinstance(data_object, Data):
                continue

            if new_property_group is not None:
                self.add_data_to_group(data_object, new_property_group)

            data_objects.append(data_object)

        # Check the depths and re-sort data if necessary
        if self.workspace.version >= 2.0:
            self.save(add_children=False)
        else:
            self.sort_depths()

        if len(data_objects) == 1:
            return data_objects[0]

        return data_objects

    def desurvey(self, depths):
        """
        Function to return x, y, z coordinates from depth.
        """
        assert (
            self.surveys is not None and self.collar is not None
        ), "'surveys' and 'collar' attributes required for desurvey operation"

        if isinstance(depths, list):
            depths = np.asarray(depths)

        ind_loc = np.maximum(
            np.searchsorted(self.surveys[:, 0], depths, side="left") - 1,
            0,
        )

        deviation_x = compute_deviation(self.surveys, "x")
        deviation_y = compute_deviation(self.surveys, "y")
        deviation_z = compute_deviation(self.surveys, "z")

        ind_dev = np.minimum(ind_loc, deviation_x.shape[0] - 1)
        locations = (
            self.locations[ind_loc, :]
            + (depths - self.surveys[ind_loc, 0])[:, None]
            * np.c_[
                deviation_x[ind_dev],
                deviation_y[ind_dev],
                deviation_z[ind_dev],
            ]
        )
        return locations

    def add_vertices(self, xyz):
        """
        Function to add vertices to the drillhole
        """
        indices = np.arange(xyz.shape[0])
        if self.n_vertices is None:
            self.vertices = xyz
        else:
            indices += self.vertices.shape[0]
            self.vertices = np.vstack([self.vertices, xyz])

        return indices.astype("uint32")

    def validate_depth_data(self, from_to, values, collocation_distance=1e-4) -> str:
        """
        Compare new and current depth values and re-use the property group if possible.
        Otherwise a new property group is added.

        :param from_to: Array of from-to values.
        :param values: Data values to be added on the from-to intervals.
        :collocation_distance: Threshold on the comparison between existing depth values.
        """
        if isinstance(from_to, list):
            from_to = np.vtack(from_to)

        assert from_to.shape[0] >= len(values), (
            f"Mismatch between input 'from_to' shape{from_to.shape} "
            + f"and 'values' shape{values.shape}"
        )
        assert from_to.shape[1] == 2, "The `from-to` values must have shape(*, 2)"

        property_group = None
        incrementer = ""
        ind = 0
        for _from, _to in zip(self._from, self._to):
            ind = len(self.parent.property_group_ids)
            incrementer = f"({ind+1})"
            if (
                _from in self.children
                and _from.values.shape[0] == from_to.shape[0]
                and np.allclose(
                    np.c_[_from.values, _to.values], from_to, atol=collocation_distance
                )
            ):
                property_group = [
                    prop_group
                    for prop_group in _from.parent.property_groups
                    if _from.uid in prop_group.properties
                ][0]

        if property_group is None:
            from_to = self.add_data(
                {
                    "FROM"
                    + incrementer: {
                        "association": "DEPTH",
                        "values": from_to[:, 0],
                        "entity_type": {"primitive_type": "FLOAT"},
                        "parent": self,
                        "allow_move": False,
                        "allow_delete": False,
                    },
                    "TO"
                    + incrementer: {
                        "association": "DEPTH",
                        "values": from_to[:, 1],
                        "entity_type": {"primitive_type": "FLOAT"},
                        "parent": self,
                        "allow_move": False,
                        "allow_delete": False,
                    },
                }
            )
            property_group = self.add_data_to_group(from_to, f"Interval_{ind+1}")

        return property_group.name

    def validate_interval_data(
        self,
        from_to: np.ndarray | list,
        values: np.ndarray,
        collocation_distance: float = 1e-4,
    ):
        """
        Compare new and current depth values, append new vertices if necessary and return
        an augmented values vector that matches the vertices indexing.
        """
        if isinstance(from_to, list):
            from_to = np.vtack(from_to)

        if from_to.shape[0] != len(values):
            raise ValueError(
                f"Mismatch between input 'from_to' shape{from_to.shape} "
                + f"and 'values' shape{values.shape}"
            )

        if from_to.shape[1] != 2:
            raise ValueError("The `from-to` values must have shape(*, 2).")

        if (self._from is None) and (self._to is None):
            uni_depth, inv_map = np.unique(from_to, return_inverse=True)
            self.cells = self.add_vertices(self.desurvey(uni_depth))[inv_map].reshape(
                (-1, 2)
            )
            self.workspace.create_entity(
                Data,
                entity={
                    "parent": self,
                    "association": "CELL",
                    "name": "FROM",
                    "values": from_to[:, 0],
                },
                entity_type={"primitive_type": "FLOAT"},
            )
            self.workspace.create_entity(
                Data,
                entity={
                    "parent": self,
                    "association": "CELL",
                    "name": "TO",
                    "values": from_to[:, 1],
                },
                entity_type={"primitive_type": "FLOAT"},
            )
        elif self.cells is not None:
            from_ind = match_values(
                self._from.values,
                from_to[:, 0],
                collocation_distance=collocation_distance,
            )
            to_ind = match_values(
                self._to.values,
                from_to[:, 1],
                collocation_distance=collocation_distance,
            )

            # Find matching cells
            in_match = np.ones((self._from.values.shape[0], 2)) * np.nan
            in_match[from_ind[:, 0], 0] = from_ind[:, 1]
            in_match[to_ind[:, 0], 1] = to_ind[:, 1]

            out_match = np.ones_like(from_to) * np.nan
            out_match[from_ind[:, 1], 0] = from_ind[:, 0]
            out_match[to_ind[:, 1], 1] = to_ind[:, 0]

            cell_map = np.c_[
                np.where(in_match[:, 0] == in_match[:, 1])[0],
                np.where(out_match[:, 0] == out_match[:, 1])[0],
            ]

            # Add vertices
            vert_new = np.ones_like(from_to, dtype="bool")
            vert_new[from_ind[:, 1], 0] = False
            vert_new[to_ind[:, 1], 1] = False
            ind_new = np.where(vert_new.flatten())[0]
            uni_new, inv_map = np.unique(
                from_to.flatten()[ind_new], return_inverse=True
            )

            # Add cells
            new_cells = np.ones_like(from_to.flatten()) * np.nan
            new_cells[ind_new] = self.add_vertices(self.desurvey(uni_new))[inv_map]
            new_cells = new_cells.reshape((-1, 2))
            new_cells[from_ind[:, 1], 0] = self.cells[from_ind[:, 0], 0]
            new_cells[to_ind[:, 1], 1] = self.cells[to_ind[:, 0], 1]
            new_cells = np.delete(new_cells, cell_map[:, 1], 0)

            # Append values
            values = merge_arrays(
                np.ones(self.n_cells) * np.nan,
                np.r_[values],
                replace="B->A",
                mapping=cell_map,
            )
            self.cells = np.r_[self.cells, new_cells.astype("uint32")]
            self._from.values = merge_arrays(
                self._from.values, from_to[:, 0], mapping=cell_map
            )
            self._to.values = merge_arrays(
                self._to.values, from_to[:, 1], mapping=cell_map
            )

        return values

    def validate_log_data(
        self,
        depth: np.ndarray,
        input_values: np.ndarray,
        collocation_distance=1e-4,
    ) -> np.ndarray:
        """
        Compare new and current depth values. Append new vertices if necessary and return
        an augmented values vector that matches the vertices indexing.
        """
        if depth.shape != input_values.shape:
            raise ValueError(
                f"Mismatch between input 'depth' shape{depth.shape} "
                + f"and 'values' shape{input_values.shape}"
            )

        input_values = np.r_[input_values]

        if self.depths is None:
            self.add_vertices(self.desurvey(depth))
            self.depths = np.r_[
                np.ones(self.n_vertices - depth.shape[0]) * np.nan, depth
            ]
            values = np.r_[
                np.ones(self.n_vertices - input_values.shape[0]) * np.nan, input_values
            ]
        else:
            depths, indices = merge_arrays(
                self.depths.values,
                depth,
                return_mapping=True,
                collocation_distance=collocation_distance,
            )
            values = merge_arrays(
                np.ones(self.n_vertices) * np.nan,
                input_values,
                replace="B->A",
                mapping=indices,
            )
            self.add_vertices(self.desurvey(np.delete(depth, indices[:, 1])))
            self.depths.values = depths

        return values

    def validate_data(self, attributes: dict, property_group=None) -> tuple:
        """
        Validate input drillhole data attributes.

        :param attributes: Dictionary of data attributes.
        :param property_group: Input property group to validate against.
        """
        collocation_distance = attributes.get(
            "collocation_distance", self.default_collocation_distance
        )
        if collocation_distance < 0:
            raise UserWarning("Input depth 'collocation_distance' must be >0.")

        if (
            "depth" not in attributes
            and "from-to" not in attributes
            and "association" not in attributes
        ):
            assert attributes["association"] == "OBJECT", (
                "Input data dictionary must contain {key:values} "
                + "{'from-to':numpy.ndarray} "
                + "or {'association': 'OBJECT'}."
            )

        if "depth" in attributes.keys():
            if self.workspace.version == 1.0:
                attributes["association"] = "VERTEX"
                attributes["values"] = self.validate_log_data(
                    attributes["depth"],
                    attributes["values"],
                    collocation_distance=collocation_distance,
                )

            else:
                attributes["from-to"] = np.c_[
                    attributes["depth"], attributes["depth"] + collocation_distance
                ]

            del attributes["depth"]

        if "from-to" in attributes.keys():
            if self.workspace.version >= 2.0:
                attributes["association"] = "DEPTH"
                property_group = self.validate_depth_data(
                    attributes["from-to"],
                    attributes["values"],
                    collocation_distance=collocation_distance,
                )
            else:
                attributes["association"] = "CELL"
                attributes["values"] = self.validate_interval_data(
                    attributes["from-to"],
                    attributes["values"],
                    collocation_distance=collocation_distance,
                )
            del attributes["from-to"]

        return attributes, property_group

    def sort_depths(self):
        """
        Read the 'DEPTH' data and sort all Data.values if needed
        """
        if self.get_data("DEPTH"):
            data_obj = self.get_data("DEPTH")[0]
            depths = data_obj.check_vector_length(data_obj.values)
            if not np.all(np.diff(depths) >= 0):
                sort_ind = np.argsort(depths)

                for child in self.children:
                    if isinstance(child, Data) and child.association.name == "VERTEX":
                        child.values = child.check_vector_length(child.values)[sort_ind]

                if self.vertices is not None:
                    self.vertices = self.vertices[sort_ind, :]

                if self.cells is not None:
                    key_map = np.argsort(sort_ind)[self.cells.flatten()]
                    self.cells = key_map.reshape((-1, 2)).astype("uint32")


def compute_deviation(surveys: np.ndarray, axis: str) -> np.ndarray | None:
    """Compute deviation from survey parameters"""
    if surveys is None:
        return None

    lengths = surveys[1:, 0] - surveys[:-1, 0]
    if axis == "x":
        dl_in = np.cos(np.deg2rad(450.0 - surveys[:-1, 2] % 360.0)) * np.cos(
            np.deg2rad(surveys[:-1, 1])
        )
        dl_out = np.cos(np.deg2rad(450.0 - surveys[1:, 2] % 360.0)) * np.cos(
            np.deg2rad(surveys[1:, 1])
        )
        ddl = np.divide(dl_out - dl_in, lengths, where=lengths != 0)

    elif axis == "y":
        dl_in = np.sin(np.deg2rad(450.0 - surveys[:-1, 2] % 360.0)) * np.cos(
            np.deg2rad(surveys[:-1, 1])
        )
        dl_out = np.sin(np.deg2rad(450.0 - surveys[1:, 2] % 360.0)) * np.cos(
            np.deg2rad(surveys[1:, 1])
        )
        ddl = np.divide(dl_out - dl_in, lengths, where=lengths != 0)

    elif axis == "z":
        dl_in = np.sin(np.deg2rad(surveys[:-1, 1]))
        dl_out = np.sin(np.deg2rad(surveys[1:, 1]))
        ddl = np.divide(dl_out - dl_in, lengths, where=lengths != 0)

    else:
        raise ValueError("Input 'axis' must be 'x', 'y' and 'z'.")
    return dl_in + lengths * ddl / 2.0
