#  Copyright (c) 2023 Mira Geoscience Ltd.
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

import numpy as np

from geoh5py.data import Data

from .entity import ConcatenatedObject
from .property_group import ConcatenatedPropertyGroup


class ConcatenatedDrillhole(ConcatenatedObject):
    @property
    def depth_(self) -> list[Data]:
        obj_list = []
        for prop_group in (
            self.property_groups if self.property_groups is not None else []
        ):
            properties = [] if prop_group.properties is None else prop_group.properties
            data = [self.get_data(child)[0] for child in properties]
            if data and "depth" in data[0].name.lower():
                obj_list.append(data[0])

        return obj_list

    @property
    def from_(self) -> list[Data]:
        """
        Depth data corresponding to the tops of the interval values.
        """
        obj_list = []
        for prop_group in (
            self.property_groups if self.property_groups is not None else []
        ):
            properties = [] if prop_group.properties is None else prop_group.properties
            data = [self.get_data(child)[0] for child in properties]
            if len(data) > 0 and "from" in data[0].name.lower():
                obj_list.append(data[0])
        return obj_list

    @property
    def to_(self) -> list[Data]:
        """
        Depth data corresponding to the bottoms of the interval values.
        """
        obj_list = []
        for prop_group in (
            self.property_groups if self.property_groups is not None else []
        ):
            data = [self.get_data(child)[0] for child in prop_group.properties]
            if len(data) > 1 and "to" in data[1].name.lower():
                obj_list.append(data[1])
        return obj_list

    def validate_data(
        self, attributes: dict, property_group=None, collocation_distance=None
    ) -> tuple:
        """
        Validate input drillhole data attributes.

        :param attributes: Dictionary of data attributes.
        :param property_group: Input property group to validate against.
        """
        if collocation_distance is None:
            collocation_distance = attributes.get(
                "collocation_distance", getattr(self, "default_collocation_distance")
            )
        if collocation_distance < 0:
            raise UserWarning("Input depth 'collocation_distance' must be >0.")

        if (
            "depth" not in attributes
            and "from-to" not in attributes
            and "association" not in attributes
        ):
            if property_group is None:
                raise AttributeError(
                    "Input data dictionary must contain {key:values} "
                    + "{'from-to':numpy.ndarray} "
                    + "or {'association': 'OBJECT'}."
                )
            attributes["from-to"] = None

        if "depth" in attributes.keys():
            values = attributes.get("values")
            attributes["association"] = "DEPTH"
            property_group = self.validate_depth_data(
                attributes.get("depth"),
                values,
                property_group=property_group,
                collocation_distance=collocation_distance,
            )

            if (
                isinstance(values, np.ndarray)
                and values.shape[0] < property_group.depth_.values.shape[0]
            ):
                attributes["values"] = np.pad(
                    values,
                    (0, property_group.depth_.values.shape[0] - len(values)),
                    constant_values=np.nan,
                )

            del attributes["depth"]

        if "from-to" in attributes.keys():
            values = attributes.get("values")
            attributes["association"] = "DEPTH"
            property_group = self.validate_interval_data(
                attributes.get("name"),
                attributes.get("from-to"),
                attributes.get("values"),
                property_group=property_group,
                collocation_distance=collocation_distance,
            )
            if (
                isinstance(values, np.ndarray)
                and values.shape[0] < property_group.from_.values.shape[0]
            ):
                attributes["values"] = np.pad(
                    values,
                    (0, property_group.from_.values.shape[0] - len(values)),
                    constant_values=np.nan,
                )

            del attributes["from-to"]

        return attributes, property_group

    def validate_depth_data(
        self,
        depth: list | np.ndarray | None,
        values: np.ndarray,
        property_group: str | ConcatenatedPropertyGroup | None = None,
        collocation_distance: float | None = None,
    ) -> ConcatenatedPropertyGroup:
        """
        :param name: Data name.
        :param depth: Sampling depths.
        :param values: Data samples to depths.
        :param property_group: Group for possibly collocated data.
        :param collocation_distance: Tolerance to determine collocated data for
            property group assignment

        :return: Augmented property group with name/values added for collocated data
            otherwise newly created property group with name/depth/values added.
        """
        if depth is not None:
            if isinstance(depth, list):
                depth = np.vstack(depth)

            if len(depth) < len(values):
                msg = f"Mismatch between input 'depth' shape{depth.shape} "
                msg += f"and 'values' shape{values.shape}"
                raise ValueError(msg)

        if depth is not None and self.property_groups is not None:
            for group in self.property_groups:
                if (
                    group.depth_ is not None
                    and group.depth_.values.shape[0] == depth.shape[0]
                    and np.allclose(
                        group.depth_.values, depth, atol=collocation_distance
                    )
                ):
                    if isinstance(property_group, str) and group.name != property_group:
                        continue

                    return group

        ind = 0
        label = ""
        if len(self.depth_) > 0:
            ind = len(self.depth_)
            label = f"({ind})"

        if property_group is None:
            property_group = f"depth_{ind}"

        if isinstance(property_group, str):
            name_id = 0
            while True:
                out_group: ConcatenatedPropertyGroup = (
                    self.find_or_create_property_group(  # type: ignore
                        name=property_group
                        if name_id == 0
                        else f"{property_group} ({name_id})",
                        association="DEPTH",
                        property_group_type="Depth table",
                    )
                )

                if (
                    out_group.depth_ is None
                    or out_group.depth_.values.shape[0] == values.shape
                ):
                    break

                name_id += 1

        else:
            out_group = property_group

        depth = getattr(self, "add_data")(
            {
                f"DEPTH{label}": {
                    "association": "DEPTH",
                    "values": depth,
                    "entity_type": {"primitive_type": "FLOAT"},
                    "parent": self,
                    "allow_move": False,
                    "allow_delete": False,
                },
            },
            out_group,
        )

        return out_group

    def validate_interval_data(
        self,
        name: str | None,
        from_to: list | np.ndarray | None,
        values: np.ndarray,
        property_group: str | ConcatenatedPropertyGroup | None = None,
        collocation_distance=1e-4,
    ) -> ConcatenatedPropertyGroup:
        """
        Compare new and current depth values and reuse the property group if possible.
        Otherwise a new property group is added.

        :param from_to: Array of from-to values.
        :param values: Data values to be added on the from-to intervals.
        :param property_group: Property group name
        :collocation_distance: Threshold on the comparison between existing depth values.
        """
        if from_to is not None:
            if isinstance(from_to, list):
                from_to = np.vstack(from_to)
                if from_to.shape[0] == 2:
                    from_to = from_to.T

            assert from_to.shape[0] >= len(values), (
                f"Mismatch between input 'from_to' shape{from_to.shape} "
                + f"and 'values' shape{values.shape}"
            )
            assert from_to.shape[1] == 2, "The `from-to` values must have shape(*, 2)"

        if (
            from_to is not None
            and property_group is None
            and self.property_groups is not None
        ):
            for p_g in self.property_groups:
                if (
                    p_g.from_ is not None
                    and p_g.from_.values.shape[0] == from_to.shape[0]
                    and np.allclose(
                        np.c_[p_g.from_.values, p_g.to_.values],
                        from_to,
                        atol=collocation_distance,
                    )
                ):
                    return p_g

        ind = 0
        label = ""
        if len(self.from_) > 0:
            ind = len(
                list(set(self.from_))
            )  # todo: from_ return the same value x time why?
            label = f"({ind})"

        if property_group is None:
            property_group = f"Interval_{ind}"

        if isinstance(property_group, str):
            out_group: ConcatenatedPropertyGroup = getattr(
                self, "find_or_create_property_group"
            )(name=property_group, association="DEPTH")
        else:
            out_group = property_group

        if out_group.from_ is not None:
            if out_group.from_.values.shape[0] != values.shape[0]:
                raise ValueError(
                    f"Input values for '{name}' with shape({values.shape[0]}) "
                    f"do not match the from-to intervals of the group '{out_group}' "
                    f"with shape({out_group.from_.values.shape[0]}). Check values or "
                    f"assign to a new property group."
                )
            return out_group

        from_to = getattr(self, "add_data")(
            {
                f"FROM{label}": {
                    "association": "DEPTH",
                    "values": from_to[:, 0],
                    "entity_type": {"primitive_type": "FLOAT"},
                    "parent": self,
                    "allow_move": False,
                    "allow_delete": False,
                },
                f"TO{label}": {
                    "association": "DEPTH",
                    "values": from_to[:, 1],
                    "entity_type": {"primitive_type": "FLOAT"},
                    "parent": self,
                    "allow_move": False,
                    "allow_delete": False,
                },
            },
            out_group,
        )

        return out_group

    def sort_depths(self):
        """Bypass sort_depths from previous version."""
