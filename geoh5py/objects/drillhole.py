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

# pylint: disable=R0902, R0904

from __future__ import annotations
from pydantic import BaseModel
import re
import uuid
import warnings
from numbers import Real

import numpy as np

from ..data import Data, FloatData, NumericData
from ..shared.utils import box_intersect, mask_by_extent, merge_arrays
from .points import Points


class row(BaseModel):
    """
    Drillhole row data.
    """
    azim: float | None = None
    dip: float | None = None
    rad: float | None = None
    uux: float | None = None
    uuy: float | None = None
    uuz: float | None = None
    xh: float | None = None
    yh: float | None = None
    zh: float | None = None
    ccx: float | None = None
    ccy: float | None = None
    ccz: float | None = None

class Drillhole(Points):
    """
    Drillhole object class defined by a collar and survey.

    :param collar: Coordinates of the drillhole.
    :param cost: Cost estimate of the drillhole.
    :param end_of_hole: End of drillhole in meters.
    :param planning: Status of the hole, defaults to 'Default'.
    :param surveys: Survey information provided as 'Depth', 'Azimuth', 'Dip'.
    :param vertices: Coordinates of the vertices.
    :param default_collocation_distance: Minimum collocation distance for matching depth on merge.
    """

    _TYPE_UID = uuid.UUID(
        fields=(0x7CAEBF0E, 0xD16E, 0x11E3, 0xBC, 0x69, 0xE4632694AA37)
    )
    __SURVEY_DTYPE = np.dtype([("Depth", "<f4"), ("Azimuth", "<f4"), ("Dip", "<f4")])
    __COLLAR_DTYPE = np.dtype([("x", float), ("y", float), ("z", float)])
    _attribute_map = Points._attribute_map.copy()
    _attribute_map.update(
        {
            "Cost": "cost",
            "Collar": "collar",
            "Planning": "planning",
            "End of hole": "end_of_hole",
        }
    )

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        collar: np.ndarray | list | None = None,
        cost: float = 0.0,
        end_of_hole: float | None = None,
        planning: str = "Default",
        surveys: np.ndarray | list | tuple | None = None,
        vertices: np.ndarray | None = None,
        default_collocation_distance: float = 1e-2,
        **kwargs,
    ):
        self._cells: np.ndarray | None = None
        self._depths: FloatData | None = None
        self._trace: np.ndarray | None = None
        self._trace_depth: np.ndarray | None = None
        self._locations = None
        self._surveys: np.ndarray | None = None

        super().__init__(
            vertices=(0.0, 0.0, 0.0) if vertices is None else vertices, **kwargs
        )

        if vertices is None:
            self._vertices = None

        self.collar = collar
        self.cost = cost
        self.default_collocation_distance = default_collocation_distance
        self.end_of_hole = end_of_hole
        self.planning = planning

        if surveys is not None:
            self.surveys = surveys

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
        if indices.dtype != "uint32":
            raise TypeError("Indices array must be of type 'uint32'")
        self._cells = indices
        self.workspace.update_attribute(self, "cells")

    @property
    def collar(self) -> np.ndarray:
        """
        Coordinates of the collar, shape(1, 3)
        """
        return self._collar

    @collar.setter
    def collar(self, value: list | np.ndarray | None):
        if value is None:
            warnings.warn(
                "No 'collar' provided. Using (0, 0, 0) as default point at the origin.",
                UserWarning,
            )
            value = (0.0, 0.0, 0.0)

        if isinstance(value, np.ndarray):
            value = value.tolist()

        if isinstance(value, str):
            value = [float(n) for n in re.findall(r"-?\d+\.\d+", value)]

        if len(value) != 3:
            raise ValueError("Origin must be a list or numpy array of len (3,).")

        value = np.asarray(tuple(value), dtype=self.__COLLAR_DTYPE)
        self._collar = value
        self._locations = None

        if self.on_file:
            self.workspace.update_attribute(self, "attributes")

            if self.trace is not None:
                self._trace = None
                self._trace_depth = None
                self.workspace.update_attribute(self, "trace")
                self.workspace.update_attribute(self, "trace_depth")

    @property
    def cost(self) -> float:
        """
        :obj:`float`: Cost estimate of the drillhole
        """
        return self._cost

    @cost.setter
    def cost(self, value: Real):
        if not isinstance(value, Real):
            raise TypeError(f"Provided cost value must be of type {float} or int.")
        self._cost = float(value)

        if self.on_file:
            self.workspace.update_attribute(self, "attributes")

    @property
    def end_of_hole(self) -> float | None:
        """
        End of drillhole in meters
        """
        return self._end_of_hole

    @end_of_hole.setter
    def end_of_hole(self, value: Real | None):
        if not isinstance(value, (int, float, type(None))):
            raise TypeError(f"Provided end_of_hole value must be of type {int}")
        self._end_of_hole = value

        if self.on_file:
            self.workspace.update_attribute(self, "attributes")

    @property
    def extent(self) -> np.ndarray | None:
        """
        Geography bounding box of the object.

        :return: shape(2, 3) Bounding box defined by the bottom South-West and
            top North-East coordinates.
        """
        if self.collar is not None:
            return (
                np.repeat(
                    np.r_[[self.collar["x"], self.collar["y"], self.collar["z"]]], 2
                )
                .reshape((-1, 2))
                .T
            )

        return None

    @property
    def locations(self) -> np.ndarray | None:
        """
        Lookup array of the well path in x, y, z coordinates.
        """
        if getattr(self, "_locations", None) is None and self.collar is not None:
            deviations = self.deviations(self.surveys)
            self._locations = self.deviations_to_xyz(deviations, self.surveys[:, 0], self.collar)

        return self._locations

    def mask_by_extent(
        self, extent: np.ndarray, inverse: bool = False
    ) -> np.ndarray | None:
        """
        Sub-class extension of :func:`~geoh5py.shared.entity.Entity.mask_by_extent`.

        Uses the collar location only.
        """
        if self.extent is None or not box_intersect(self.extent, extent):
            return None

        if self.collar is not None:
            return mask_by_extent(
                np.c_[[self.collar["x"], self.collar["y"], self.collar["z"]]].T,
                extent,
                inverse=inverse,
            )

        return None

    @property
    def n_cells(self) -> int | None:
        """
        Number of cells.
        """
        if self.cells is not None:
            return self.cells.shape[0]
        return None

    @property
    def planning(self) -> str:
        """
        Status of the hole on of "Default", "Ongoing", "Planned", "Completed" or "No status"
        """
        return self._planning

    @planning.setter
    def planning(self, value: str):
        choices = ["Default", "Ongoing", "Planned", "Completed", "No status"]
        if value not in choices:
            raise ValueError(f"Provided planning value must be one of {choices}")
        self._planning = value

        if self.on_file:
            self.workspace.update_attribute(self, "attributes")

    @property
    def surveys(self) -> np.ndarray:
        """
        Coordinates of the surveys.
        """
        if self._surveys is None and self.on_file:
            self._surveys = self.workspace.fetch_array_attribute(self, "surveys")

        if self._surveys is None:
            surveys = np.c_[0, 0, -90]

        else:
            surveys = np.vstack(
                [
                    self._surveys["Depth"],
                    self._surveys["Azimuth"],
                    self._surveys["Dip"],
                ]
            ).T

        return surveys.astype(float)

    @surveys.setter
    def surveys(self, array: np.ndarray | list | tuple):
        if not isinstance(array, (np.ndarray, list, tuple)):
            raise TypeError(
                "Input 'surveys' must be of type 'numpy.ndarray' or 'list'."
            )

        self._surveys = self.format_survey_values(array)
        self.end_of_hole = float(self._surveys["Depth"][-1])

        if self.on_file:
            self.workspace.update_attribute(self, "surveys")

            if self.trace is not None:
                self._trace = None
                self.workspace.update_attribute(self, "trace")

        self._locations = None

    def format_survey_values(self, values: list | tuple | np.ndarray) -> np.ndarray:
        """
        Reformat the survey values as structured array with the right shape.
        """
        if isinstance(values, (list, tuple)):
            values = np.array(values, ndmin=2)

        if np.issubdtype(values.dtype, np.number):
            if values.shape[1] != 3:
                raise ValueError("'surveys' requires an ndarray of shape (*, 3)")

            array_values = np.asarray(
                np.core.records.fromarrays(values.T, dtype=self.__SURVEY_DTYPE)
            )
        else:
            array_values = values

        if array_values.dtype.descr[:3] != self.__SURVEY_DTYPE.descr:
            raise ValueError(
                f"Array of 'survey' must be of dtype = {self.__SURVEY_DTYPE}"
            )
        return array_values

    @property
    def default_collocation_distance(self):
        """
        Minimum collocation distance for matching depth on merge
        """
        return self._default_collocation_distance

    @default_collocation_distance.setter
    def default_collocation_distance(self, tol):
        if not tol > 0:
            raise ValueError("Tolerance value should be >0.")

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
    def from_(self):
        """
        Depth data corresponding to the tops of the interval values.
        """
        data_obj = self.get_data("FROM")
        if data_obj:
            return data_obj[0]

        return None

    @property
    def to_(self):
        """
        Depth data corresponding to the bottoms of the interval values.
        """
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

    def desurvey(self, depths):
        """
        Function to return x, y, z coordinates from depth.
        """
        deviations = self.deviations(self.surveys, self.collar.tolist())
        return self.deviations_to_xyz(deviations, depths)

    def add_vertices(self, xyz):
        """
        Function to add vertices to the drillhole
        """
        indices = np.arange(xyz.shape[0])
        if self._vertices is not None:
            indices += self.n_vertices
            xyz = np.vstack([self.vertices, xyz])

        self._vertices = np.asarray(
            np.core.records.fromarrays(
                xyz.T.tolist(),
                dtype=[("x", "<f8"), ("y", "<f8"), ("z", "<f8")],
            )
        )
        self.workspace.update_attribute(self, "vertices")

        return indices.astype("uint32")

    def validate_interval_data(  # pylint: disable=too-many-locals
        self,
        from_to: np.ndarray | list,
        values: np.ndarray,
        collocation_distance: float = 1e-4,
    ) -> np.ndarray:
        """
        Compare new and current depth values, append new vertices if necessary and return
        an augmented values vector that matches the vertices indexing.

        :param from_to: Array of from-to values.
        :param values: Array of values.
        :param collocation_distance: Minimum collocation distance for matching.

        :return: Augmented values vector that matches the vertices indexing.
        """
        if isinstance(from_to, list):
            from_to = np.vstack(from_to)

        if from_to.shape[0] != len(values):
            raise ValueError(
                f"Mismatch between input 'from_to' shape{from_to.shape} "
                + f"and 'values' shape{values.shape}"
            )

        if from_to.ndim != 2 or from_to.shape[1] != 2:
            raise ValueError("The `from-to` values must have shape(*, 2).")

        if (self.from_ is None) and (self.to_ is None):
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
        elif self.cells is not None and self.from_ is not None and self.to_ is not None:
            values = np.r_[values]
            out_vec = np.c_[self.from_.values, self.to_.values]
            dist_match = []
            for i, elem in enumerate(from_to):
                ind = np.where(
                    np.linalg.norm(elem - out_vec, axis=1) < collocation_distance
                )[0]
                if len(ind) > 0:
                    dist_match.append([ind[0], i])

            cell_map: np.ndarray = np.asarray(dist_match, dtype=int)

            # Add vertices
            vert_new = np.ones_like(from_to, dtype="bool")
            if cell_map.ndim == 2:
                vert_new[cell_map[:, 1], :] = False
            ind_new = np.where(vert_new.flatten())[0]
            uni_new, inv_map = np.unique(
                from_to.flatten()[ind_new], return_inverse=True
            )

            # check if its text data, and defined nan array if so
            if values.dtype.kind in ["U", "S"]:
                nan_values = np.array([""] * self.n_cells)  # type: ignore
            else:
                nan_values = np.ones(self.n_cells) * np.nan

            # Append values
            values = merge_arrays(
                nan_values,
                values,
                replace="B->A",
                mapping=cell_map,
            )
            self.cells = np.r_[
                self.cells,
                self.add_vertices(self.desurvey(uni_new))[inv_map]
                .reshape((-1, 2))
                .astype("uint32"),
            ]
            self.from_.values = merge_arrays(
                self.from_.values, from_to[:, 0], mapping=cell_map
            )
            self.to_.values = merge_arrays(
                self.to_.values, from_to[:, 1], mapping=cell_map
            )

        return values

    def validate_depth_data(
        self,
        depth: np.ndarray,
        values: np.ndarray,
        collocation_distance=1e-4,
    ) -> np.ndarray:
        """
        Compare new and current depth values. Append new vertices if necessary and return
        an augmented values vector that matches the vertices indexing.

        :param depth: Array of depth values.
        :param values: Array of values.
        :param collocation_distance: Minimum collocation distance for matching.

        :return: Augmented values vector that matches the vertices indexing.
        """
        if depth.shape != values.shape:
            raise ValueError(
                f"Mismatch between input 'depth' shape{depth.shape} "
                + f"and 'values' shape{values.shape}"
            )

        input_values = np.r_[values]

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

    def validate_association(
        self, attributes: dict, property_group=None, collocation_distance=None, **_
    ) -> tuple:
        """
        Validate input drillhole data attributes.

        :param attributes: Dictionary of data attributes.
        :param property_group: Input property group to validate against.
        :param collocation_distance: Minimum collocation distance for matching.
        """
        if collocation_distance is None:
            collocation_distance = attributes.get(
                "collocation_distance", self.default_collocation_distance
            )

        if collocation_distance < 0:
            raise UserWarning("Input depth 'collocation_distance' must be >0.")

        if "depth" not in attributes and "from-to" not in attributes:
            if "association" not in attributes or attributes["association"] != "OBJECT":
                raise ValueError(
                    "Input data dictionary must contain a key/value pair of depth data "
                    "or contain an 'OBJECT' association. Valid depth keys are 'depth' "
                    "and 'from-to'."
                )

        if attributes["name"] in self.get_data_list():
            raise ValueError(
                f"Data with name '{attributes['name']}' already present "
                f"on the drillhole '{self.name}'. "
                "Consider changing the values or renaming."
            )

        if "depth" in attributes.keys():
            attributes["association"] = "VERTEX"
            attributes["values"] = self.validate_depth_data(
                attributes["depth"],
                attributes["values"],
                collocation_distance=collocation_distance,
            )
            del attributes["depth"]

        if "from-to" in attributes.keys():
            attributes["association"] = "CELL"
            attributes["values"] = self.validate_interval_data(
                attributes["from-to"],
                attributes["values"],
                collocation_distance=collocation_distance,
            )
            del attributes["from-to"]

        return attributes, property_group

    @property
    def vertices(self) -> np.ndarray | None:
        """
        :obj:`~geoh5py.objects.object_base.ObjectBase.vertices`
        """
        if self._vertices is None and self.on_file:
            self._vertices = self.workspace.fetch_array_attribute(self, "vertices")

        if self._vertices is not None:
            return self._vertices.view("<f8").reshape((-1, 3))

        return None

    @vertices.setter
    def vertices(self, vertices: np.ndarray | list | tuple):
        xyz = self.validate_vertices(vertices)
        if self._vertices is not None and self._vertices.shape != xyz.shape:
            raise ValueError(
                "New vertices array must have the same shape as the current vertices array."
            )
        self._vertices = xyz

        self.workspace.update_attribute(self, "vertices")

    def post_processing(self):
        """
        Read the 'DEPTH' data and sort all Data.values if needed
        """
        if self.get_data("DEPTH"):
            data_obj = self.get_data("DEPTH")[0]

            depths = data_obj.values
            if isinstance(data_obj, NumericData):
                depths = data_obj.validate_values(depths)

            if not np.all(np.diff(depths) >= 0):
                sort_ind = np.argsort(depths)

                for child in self.children:
                    if (
                        isinstance(child, NumericData)
                        and getattr(child.association, "name", None) == "VERTEX"
                    ):
                        child.values = child.validate_values(child.values)[sort_ind]

                if self.vertices is not None:
                    self._vertices = self.vertices[sort_ind, :]
                    self.workspace.update_attribute(self, "vertices")

                if self.cells is not None:
                    key_map = np.argsort(sort_ind)[self.cells.flatten()]
                    self._cells = key_map.reshape((-1, 2)).astype("uint32")
                    self.workspace.update_attribute(self, "cells")

    @staticmethod
    def deviations(survey: np.ndarray, collar) -> dict:
        # Survey['Depth', 'Azimuth', 'Dip']
        # Repeat first survey point at surface for de-survey interpolation
        if survey[0, 0] != 0.0:
            full_survey = np.vstack([survey[0, :], survey])
            full_survey[0, 0] = 0.0
        else:
            full_survey = survey.copy()

        ind = np.argsort(full_survey[:, 0])
        INFINITE_RADIUS = 99999.9
        depths = full_survey[ind, 0]
        azimuth = np.deg2rad(90 - full_survey[ind, 1])
        dip = np.deg2rad(full_survey[ind, 2])

        # intervals = {
        #     depths[i]: row(azim=azimuth[i], dip=dip[i]) for i in ind
        # }
        # # for i in range(len(full_surveys)):
        # #     survey = full_surveys[i]
        # #     # adjust az for grid azimuth and geo->cart coord system
        # #     az = 90 - survey.azim
        # #     intervals[survey.depth] = IBHInterval(survey.depth, az, survey.dip)
        #
        # rd = az = di = sn = cs = dp0 = dp1 = cx0 = cy0 = cz0 = cx1 = cy1 = cz1 = vx = vy = vz = vr = ux = uy = uz = ur = 0.0
        # scp = ddp = alpha = x0 = y0 = z0 = x1 = y1 = z1 = 0.0
        # tdepth = prevdepth = 0.0
        # it_bh = iter(intervals.items())
        #
        # if intervals:
        #     it_bh = iter(intervals.items())
        #     key, value = next(it_bh)
        #     tdepth = prevdepth = dp1 = abs(key)
        #     az = value.azim
        #     di = value.dip
        #
        #     value.xh = collar[0]
        #     value.yh = collar[1]
        #     value.zh = collar[2]
        # else:
        #     tdepth = prevdepth = dp1 = 0
        #     az = 0.0
        #     di = np.deg2rad(-90.0)
        #
        # cs = np.cos(di)
        # sn = np.sin(di)
        # cx1 = cs * np.cos(az)
        # cy1 = cs * np.sin(az)
        # cz1 = sn
        # if len(intervals) > 1:
        #     it_bh = iter(intervals.items())
        #     next(it_bh)
        #     for key, value in it_bh:
        #         tdepth = key
        #         az = value.azim
        #         di = value.dip
        #         dp0 = dp1
        #         dp1 = abs(tdepth)
        #         ddp = dp1 - dp0
        #         if ddp == 0.0:
        #             ddp = 1.0
        #         cx0 = cx1
        #         cy0 = cy1
        #         cz0 = cz1
        #         cs = np.cos(di)
        #         sn = np.sin(di)
        #         cx1 = cs * np.cos(az)
        #         cy1 = cs * np.sin(az)
        #         cz1 = sn
        #         # v = cross product between c0 and c1
        #         vx = cy0 * cz1 - cz0 * cy1
        #         vy = cz0 * cx1 - cx0 * cz1
        #         vz = cx0 * cy1 - cy0 * cx1
        #         vr = np.sqrt(vx * vx + vy * vy + vz * vz)
        #         # scp = dot product between c0 and c1
        #         scp = cx0 * cx1 + cy0 * cy1 + cz0 * cz1
        #         ux = cx1 - scp * cx0
        #         uy = cy1 - scp * cy0
        #         uz = cz1 - scp * cz0
        #         ur = np.sqrt(ux * ux + uy * uy + uz * uz)
        #         if ur == 0.0:
        #             ur = INFINITE_RADIUS
        #         ux = ux / ur
        #         uy = uy / ur
        #         uz = uz / ur
        #
        #         # scp/vr = cos(gamma)/sin(gamma) = cot(gamma) with gamma the angle between c0 and c1.
        #         # arccot(x) = pi/2 - arctan(x)
        #         # (https://en.wikipedia.org/wiki/Inverse_trigonometric_functions) so alpha = abs(pi/2 -
        #         # arctan(scp/vr)) = abs(arccot(scp/vr)) = abs(arccot(cot(gamma))) = abs(gamma) alpha is
        #         # the positive angle between c0 and c1.
        #         alpha = abs(0.5 * np.pi - np.arctan2(scp, vr))
        #         # {should range from 0 to pi}
        #         if alpha != 0.0:
        #             # length of an arc of circle (here ddp) = radius * angle,
        #             # so we approximate the path between 2 stations by an arc of circle.
        #             rd = ddp / alpha
        #         else:
        #             rd = ddp * INFINITE_RADIUS  # {flag infinite radius}
        #             alpha = ddp / rd
        #         intervals[prevdepth].rad = rd
        #         intervals[prevdepth].uux = ux
        #         intervals[prevdepth].uuy = uy
        #         intervals[prevdepth].uuz = uz
        #         intervals[prevdepth].ccx = cx0
        #         intervals[prevdepth].ccy = cy0
        #         intervals[prevdepth].ccz = cz0
        #         x0 = intervals[prevdepth].xh
        #         y0 = intervals[prevdepth].yh
        #         z0 = intervals[prevdepth].zh
        #         sn = np.sin(alpha)
        #         cs = 1.0 - np.cos(alpha)
        #         x1 = x0 + rd * (cx0 * sn + ux * cs)
        #         y1 = y0 + rd * (cy0 * sn + uy * cs)
        #         z1 = z0 + rd * (cz0 * sn + uz * cs)
        #         intervals[tdepth].xh = x1
        #         intervals[tdepth].yh = y1
        #         intervals[tdepth].zh = z1
        #         prevdepth = tdepth
        #
        #     intervals[tdepth].uux = 0.0
        #     intervals[tdepth].uuy = 0.0
        #     intervals[tdepth].uuz = 0.0
        #     intervals[tdepth].rad = INFINITE_RADIUS
        #     intervals[tdepth].ccx = cx1
        #     intervals[tdepth].ccy = cy1
        #     intervals[tdepth].ccz = cz1
        # else:
        #     if 0.0 not in intervals:
        #         intervals[0.0] = IBHInterval(tdepth, az, di)
        #     intervals[0.0].rad = INFINITE_RADIUS
        #     intervals[0.0].ccx = cx1
        #     intervals[0.0].ccy = cy1
        #     intervals[0.0].ccz = cz1
        #     intervals[0.0].uux = 0.0
        #     intervals[0.0].uuy = 0.0
        #     intervals[0.0].uuz = 0.0
        #     intervals[0.0].xh = collar[0]
        #     intervals[0.0].yh = collar[1]
        #     intervals[0.0].zh = collar[2]

        sin_dip = np.sin(dip)
        cos_dip = np.cos(dip)

        dx = cos_dip * np.cos(azimuth)
        dy = cos_dip * np.sin(azimuth)
        dz = sin_dip

        if survey.shape[0] < 2:
            return {
                "depths": 0,
                "rad": np.r_[INFINITE_RADIUS],
                "uux": np.r_[0],
                "uuy": np.r_[0],
                "uuz": np.r_[0],
                "dx": dx,
                "dy": dy,
                "dz": dz,
                "x": collar[0],
                "y": collar[1],
                "z": collar[2]
            }

        # Cross product between neighbours
        cross = np.cross(np.c_[dx[:-1], dy[:-1], dz[:-1]], np.c_[dx[1:], dy[1:], dz[1:]], axis=1)
        vr = np.linalg.norm(cross, axis=1)

        scp = np.sum(np.c_[dx[:-1], dy[:-1], dz[:-1]] * np.c_[dx[1:], dy[1:], dz[1:]], axis=1)
        ux = dx[1:] - scp * dx[:-1]
        uy = dy[1:] - scp * dy[:-1]
        uz = dz[1:] - scp * dz[:-1]
        ur = np.linalg.norm(np.c_[ux, uy, uz], axis=1)
        ur[ur == 0.0] = INFINITE_RADIUS

        ux /= ur
        uy /= ur
        uz /= ur

        alpha = np.abs(0.5 * np.pi - np.arctan2(scp, vr))
        ddp = np.diff(depths)
        rd = ddp / alpha

        rd[alpha == 0.0] = ddp[alpha == 0.0] * INFINITE_RADIUS
        alpha[alpha == 0.0] = ddp[alpha == 0.0] / rd[alpha == 0.0]

        intervals = {
            "depths": full_survey[:, 0],
            "rad": np.r_[rd, INFINITE_RADIUS],
            "uux": np.r_[ux, 0],
            "uuy": np.r_[uy, 0],
            "uuz": np.r_[uz, 0],
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "x": np.r_[collar[0], collar[0] + np.cumsum(rd * (dx[:-1] * np.sin(alpha) + ux * (1 - np.cos(alpha))))],
            "y": np.r_[collar[1], collar[1] + np.cumsum(rd * (dy[:-1] * np.sin(alpha) + uy * (1 - np.cos(alpha))))],
            "z": np.r_[collar[2], collar[2] + np.cumsum(rd * (dz[:-1] * np.sin(alpha) + uz * (1 - np.cos(alpha))))],
        }

        return intervals

    @staticmethod
    def deviations_to_xyz(deviations: dict, depths: np.ndarray) -> np.ndarray:
        """
        Convert survey parameters to x, y, z coordinates.

        :param deviations: Dictionary of deviation parameters.
        :param depths: Array of depth values.
        :param collar: Array of collar coordinates.
        """
        ind = np.searchsorted(deviations["depths"], depths, side="right") - 1
        dl = depths - deviations["depths"][ind]
        radii = deviations["rad"][ind]
        radii[radii == 0.0] = 1.0

        angle = dl / radii

        x = deviations["x"][ind] + radii * (
            deviations["dx"][ind] * np.sin(angle) + deviations["uux"][ind] * (1 - np.cos(angle))
        )
        y = deviations["y"][ind] + radii * (
            deviations["dy"][ind] * np.sin(angle) + deviations["uuy"][ind] * (1 - np.cos(angle))
        )
        z = deviations["z"][ind] + radii * (
            deviations["dz"][ind] * np.sin(angle) + deviations["uuz"][ind] * (1 - np.cos(angle))
        )

        return np.c_[x, y, z]


def compute_deviation(surveys: np.ndarray) -> np.ndarray:
    """
    Compute deviation distances from survey parameters.

    :param surveys: Array of azimuth, dip and depth values.
    """
    if surveys is None:
        raise AttributeError(
            "Cannot compute deviation coordinates without `survey` attribute."
        )

    lengths = surveys[1:, 0] - surveys[:-1, 0]

    deviation = []
    for component in [deviation_x, deviation_y, deviation_z]:
        dl_in = component(surveys[:-1, 1], surveys[:-1, 2])
        dl_out = component(surveys[1:, 1], surveys[1:, 2])
        ddl = np.divide(dl_out - dl_in, lengths, where=lengths != 0)
        deviation += [dl_in + lengths * ddl / 2.0]

    return np.vstack(deviation).T


def deviation_x(azimuth, dip):
    """
    Compute the easting deviation.

    :param azimuth: Degree angle clockwise from North
    :param dip: Degree angle positive down from horizontal

    :return deviation: Change in easting distance.
    """
    return np.cos(np.deg2rad(450.0 - azimuth % 360.0)) * np.cos(np.deg2rad(dip))


def deviation_y(azimuth, dip):
    """
    Compute the northing deviation.

    :param azimuth: Degree angle clockwise from North
    :param dip: Degree angle positive down from horizontal

    :return deviation: Change in northing distance.
    """
    return np.sin(np.deg2rad(450.0 - azimuth % 360.0)) * np.cos(np.deg2rad(dip))


def deviation_z(_, dip):
    """
    Compute the vertical deviation.

    :param dip: Degree angle positive down from horizontal

    :return deviation: Change in vertical distance.
    """
    return np.sin(np.deg2rad(dip))


