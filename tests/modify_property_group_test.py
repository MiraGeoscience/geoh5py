# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2020-2026 Mira Geoscience Ltd.                                     '
#                                                                              '
#  This file is part of geoh5py.                                               '
#                                                                              '
#  geoh5py is free software: you can redistribute it and/or modify             '
#  it under the terms of the GNU Lesser General Public License as published by '
#  the Free Software Foundation, either version 3 of the License, or           '
#  (at your option) any later version.                                         '
#                                                                              '
#  geoh5py is distributed in the hope that it will be useful,                  '
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              '
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               '
#  GNU Lesser General Public License for more details.                         '
#                                                                              '
#  You should have received a copy of the GNU Lesser General Public License    '
#  along with geoh5py.  If not, see <https://www.gnu.org/licenses/>.           '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


from __future__ import annotations

from abc import ABC

import numpy as np
import pytest

from geoh5py.objects import Curve
from geoh5py.workspace import Workspace


def test_modify_property_group(tmp_path):
    def compare_objects(object_a, object_b, ignore=None):
        if ignore is None:
            ignore = ["_workspace", "_children", "_parent", "_property_table"]

        for attr in object_a.__dict__.keys():
            if attr in ignore:
                continue
            if isinstance(getattr(object_a, attr[1:]), ABC):
                compare_objects(
                    getattr(object_a, attr[1:]), getattr(object_b, attr[1:])
                )
            else:
                assert np.all(
                    getattr(object_a, attr[1:]) == getattr(object_b, attr[1:])
                ), (
                    f"Output attribute {attr[1:]} for {object_a} do not match input {object_b}"
                )

    obj_name = "myCurve"
    # Generate a curve with multiple data
    xyz = np.c_[np.linspace(0, 2 * np.pi, 12), np.zeros(12), np.zeros(12)]
    h5file_path = tmp_path / r"prop_group_test.geoh5"
    with Workspace.create(h5file_path) as workspace:
        curve = Curve.create(workspace, vertices=xyz, name=obj_name)

        assert curve.property_groups is None
        # Add data
        props = []
        for i in range(4):
            values = np.cos(xyz[:, 0] / (i + 1))
            props += [
                curve.add_data(
                    {f"Period{i + 1}": {"values": values}}, property_group="myGroup"
                )
            ]

        children_list = curve.get_data_list()
        assert all(f"Period{i + 1}" in children_list for i in range(4)), (
            "Missing data children"
        )
        # Property group object should have been created
        prop_group = curve.fetch_property_group(name="myGroup")

        # Remove on props from the list
        prop_group.remove_properties(curve.children[0])
        prop_group.remove_properties(props[-2:])

        assert len(prop_group.properties) == 1, "Error removing a property_group"

        # Add a CELL property group
        curve.add_data(
            {"Cell_values": {"values": np.ones(curve.n_cells), "association": "CELL"}},
            property_group="cell_group",
        )

        assert (
            curve.fetch_property_group(name="cell_group").association.name == "CELL"
        ), "Failed to create a CELL property_group"

        # Re-open the workspace
        workspace = Workspace(h5file_path)

        # Read the property_group back in
        rec_curve = workspace.get_entity(obj_name)[0]
        rec_prop_group = rec_curve.fetch_property_group(name="myGroup")

        compare_objects(rec_prop_group, prop_group)

        with pytest.raises(DeprecationWarning):
            workspace.fetch_property_groups(rec_curve)

        compare_objects(rec_curve.fetch_property_group(name="myGroup"), prop_group)

        assert (
            rec_curve.fetch_property_group(name="cell_group").association.name == "CELL"
        ), "Failed to recover a CELL property_group"
