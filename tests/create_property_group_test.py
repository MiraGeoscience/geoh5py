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

from os import path

import numpy as np

from geoh5py.objects import Curve
from geoh5py.workspace import Workspace


def test_create_property_group(tmp_path):
    h5file_path = path.join(tmp_path, "prop_group_test.geoh5")

    with Workspace(h5file_path) as workspace:
        curve = Curve.create(
            workspace,
            vertices=np.c_[np.linspace(0, 2 * np.pi, 12), np.zeros(12), np.zeros(12)],
        )
        # Add data
        props = []
        for i in range(4):
            values = np.cos(curve.vertices[:, 0] / (i + 1))
            props += [
                curve.add_data(
                    {f"Period{i+1}": {"values": values}}, property_group="myGroup"
                )
            ]

        # Property group object should have been created
        prop_group = curve.find_or_create_property_group(name="myGroup")
        # Create a new group by data name
        single_data_group = curve.add_data_to_group(f"Period{1}", "Singleton")

        assert (
            workspace.find_data(single_data_group.properties[0]).name == f"Period{1}"
        ), "Failed at creating a property group by data name"

        # Create a new group by data uid
        single_data_group = curve.add_data_to_group(props[1].uid, "Singleton")

        assert (
            workspace.find_data(single_data_group.properties[0]).name == f"Period{1}"
        ), "Failed at creating a property group by data name"

        # Re-open the workspace
        workspace = Workspace(h5file_path)
        rec_object = workspace.get_entity(curve.uid)[0]
        # Read the property_group back in
        rec_prop_group = rec_object.find_or_create_property_group(name="myGroup")

        attrs = rec_prop_group.attribute_map
        check_list = [
            attr
            for attr in attrs.values()
            if getattr(rec_prop_group, attr) != getattr(prop_group, attr)
        ]
        assert (
            len(check_list) == 0
        ), f"Attribute{check_list} of PropertyGroups in output differ from input"

        # Copy an object without children
        new_curve = rec_object.copy(copy_children=False)

        assert (
            new_curve.property_groups == []
        ), "Property_groups not properly removed on copy without children."
