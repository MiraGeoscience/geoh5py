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

from uuid import uuid4

import numpy as np
import pytest

from geoh5py.groups import PropertyGroup
from geoh5py.objects import Curve
from geoh5py.workspace import Workspace


def test_create_property_group(tmp_path):
    #  pylint: disable=too-many-locals
    # pylint: disable=too-many-statements

    h5file_path = tmp_path / r"prop_group_test.geoh5"

    with Workspace.create(h5file_path) as workspace:
        curve = Curve.create(
            workspace,
            vertices=np.c_[np.linspace(0, 2 * np.pi, 12), np.zeros(12), np.zeros(12)],
        )

        # Add data
        props = []
        test_values = []
        for i in range(4):
            values = np.cos(curve.vertices[:, 0] / (i + 1))
            props += [
                curve.add_data(
                    {f"Period{i+1}": {"values": values}}, property_group="myGroup"
                )
            ]
            test_values.append(values)

        with pytest.raises(TypeError, match="Name must be"):
            _ = PropertyGroup(parent="bidon", name=42)

        with pytest.raises(AttributeError, match="Parent bidon"):
            _ = PropertyGroup(parent="bidon")

        # Property group object should have been created
        prop_group = curve.find_or_create_property_group(name="myGroup")

        # test properties group
        curve2 = curve.copy()

        prop_group2 = curve2.find_or_create_property_group(name="myGroup2")

        assert prop_group2.collect_values is None

        _ = curve2.copy()

        assert prop_group2.remove_properties("bidon") is None

        with pytest.raises(TypeError, match="All uids must be of type"):
            prop_group2.properties = [123]

        with pytest.raises(TypeError, match="Could not convert input uid"):
            prop_group2.uid = 123

        with pytest.raises(TypeError, match="Attribute 'on_file' must be a boolean"):
            prop_group.on_file = "bidon"

        with pytest.raises(KeyError, match="A Property Group with name"):
            curve.create_property_group(name="myGroup")

        # test error for allow delete
        with pytest.raises(TypeError, match="allow_delete must be a boolean"):
            prop_group.allow_delete = "bidon"

        prop_group.allow_delete = False
        assert prop_group.allow_delete is False

        # set parent
        assert prop_group.parent == curve

        # Add data to group as object
        single_data_group = curve.add_data_to_group(props[1], "Singleton")

        assert (
            len(single_data_group.properties) == 1  # 2
        ), "Failed adding data to property group."

        # Add data to group by uid
        single_data_group.add_properties(props[2].uid)

        assert (
            len(single_data_group.properties) == 2  # 3
        ), "Failed adding data to property group."

        with pytest.raises(UserWarning, match="Cannot modify"):
            single_data_group.properties = "bidon"

        # Try adding bogus data on group
        single_data_group.add_properties(uuid4())
        assert len(single_data_group.properties) == 2  # 3

        # Try adding a data that doesn't belong
        single_data_group.add_properties(curve2.children[0])
        assert len(single_data_group.properties) == 2

        # Remove data from group by data
        single_data_group.remove_properties(props[2])
        assert len(single_data_group.properties) == 1  # 2

        # Remove bogus data from uuid
        single_data_group.remove_properties(uuid4())
        assert len(single_data_group.properties) == 1  # 2

        # Remove data that doesn't belong
        single_data_group.remove_properties(curve2.children[0])
        assert len(single_data_group.properties) == 1  # 2

        # get property group
        property_group_test = workspace.get_entity("myGroup")[0]
        assert isinstance(property_group_test, PropertyGroup)

        assert isinstance(workspace.property_groups, list)

        with pytest.raises(
            TypeError, match="property_group must be a PropertyGroup instance"
        ):
            workspace.register_property_group("bidon")

        property_group_from_object = curve.get_entity("myGroup")[0]

        assert property_group_from_object == property_group_test

        np.testing.assert_almost_equal(
            property_group_from_object.collect_values, np.array(test_values)
        )

    # Re-open the workspace
    with Workspace(h5file_path) as workspace:
        # problem with this line with some ubuntu and macos versions
        # assert workspace.get_entity("myGroup")[0].uid == property_group_test.uid

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
            new_curve.property_groups is None
        ), "Property_groups not properly removed on copy without children."

        #
        rec_object.remove_children(rec_prop_group)
        assert len(rec_object.property_groups) == 1, "Failed to remove property group"

        rec_object.remove_children(rec_object.property_groups[0])

    with Workspace(h5file_path) as workspace:
        rec_object = workspace.get_entity(curve.uid)[0]
        assert (
            rec_object.property_groups is None
        ), "Property_groups not properly removed."


def test_copy_property_group(tmp_path):
    h5file_path = tmp_path / r"prop_group_test.geoh5"

    with Workspace.create(h5file_path) as workspace:
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

        # New property group object should have been created on copy
        curve_2 = curve.copy()

        assert len(curve_2.property_groups) > 0
        assert all(
            len(curve_2.get_data(uid)) > 0
            for uid in curve_2.property_groups[0].properties
        )
