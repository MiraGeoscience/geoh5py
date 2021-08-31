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

import tempfile
from abc import ABC
from pathlib import Path

import numpy as np

from geoh5py.objects import Curve
from geoh5py.workspace import Workspace


def test_modify_property_group():
    def compare_objects(object_a, object_b, ignore=None):
        if ignore is None:
            ignore = ["_workspace", "_children", "_parent"]
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
                ), f"Output attribute {attr[1:]} for {object_a} do not match input {object_b}"

    obj_name = "myCurve"
    # Generate a curve with multiple data
    n_stn = 12
    xyz = np.c_[np.linspace(0, 2 * np.pi, n_stn), np.zeros(n_stn), np.zeros(n_stn)]

    with tempfile.TemporaryDirectory() as tempdir:
        h5file_path = Path(tempdir) / r"prop_group_test.geoh5"

        # Create a workspace
        workspace = Workspace(h5file_path)

        curve = Curve.create(workspace, vertices=xyz, name=obj_name)

        # Add data
        props = []
        for i in range(4):
            values = np.cos(xyz[:, 0] / (i + 1))
            props += [
                curve.add_data(
                    {f"Period{i+1}": {"values": values}}, property_group="myGroup"
                )
            ]

        children_list = curve.get_data_list()
        assert all(
            f"Period{i + 1}" in children_list for i in range(4)
        ), "Missing data children"
        # Property group object should have been created
        prop_group = curve.find_or_create_property_group(name="myGroup")

        # Remove on props from the list
        curve.remove_data_from_group(children_list[0], name="myGroup")
        curve.remove_data_from_group(props[-2:], name="myGroup")

        assert len(prop_group.properties) == 1, "Error removing a property_group"

        workspace.finalize()

        # Re-open the workspace
        workspace = Workspace(h5file_path)

        # Read the property_group back in
        rec_curve = workspace.get_entity(obj_name)[0]
        rec_prop_group = rec_curve.find_or_create_property_group(name="myGroup")

        compare_objects(rec_prop_group, prop_group)
