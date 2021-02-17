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

import tempfile
from pathlib import Path

import numpy as np

from geoh5py.objects import Curve
from geoh5py.workspace import Workspace


def test_create_property_group():

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
        for i in range(4):
            values = np.cos(xyz[:, 0] / (i + 1))
            curve.add_data(
                {f"Period{i+1}": {"values": values}}, property_group="myGroup"
            )

        # Property group object should have been created
        prop_group = curve.get_property_group("myGroup")

        # Create a new group by data name
        single_data_group = curve.add_data_to_group(f"Period{1}", "Singleton")

        assert (
            workspace.find_data(single_data_group.properties[0]).name == f"Period{1}"
        ), "Failed at creating a property group by data name"
        workspace.finalize()

        # Re-open the workspace
        workspace = Workspace(h5file_path)

        # Read the property_group back in
        rec_prop_group = workspace.get_entity(obj_name)[0].get_property_group("myGroup")

        attrs = rec_prop_group.attribute_map
        check_list = [
            attr
            for attr in attrs.values()
            if getattr(rec_prop_group, attr) != getattr(prop_group, attr)
        ]
        assert (
            len(check_list) == 0
        ), f"Attribute{check_list} of PropertyGroups in output differ from input"
