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

import numpy as np
import pytest

from geoh5py.data import Data, DataAssociationEnum
from geoh5py.groups import PropertyGroup
from geoh5py.groups.property_group import GroupTypeEnum
from geoh5py.groups.property_group_table import PropertyGroupTable
from geoh5py.objects import Curve, Drillhole
from geoh5py.workspace import Workspace


def make_example(workspace, add_str_column=False):
    curve = Curve.create(
        workspace,
        vertices=np.c_[np.linspace(0, 2 * np.pi, 12), np.zeros(12), np.zeros(12)],
        name="curve",
    )

    # Add data
    props = {}
    all_values = []
    for i in range(4):
        values = np.cos(curve.vertices[:, 0] / (i + 1))
        props[f"Period{i + 1}"] = {"values": values}
        all_values.append(values)

    if add_str_column:
        values = np.array(["i" for i in range(12)])
        props["StrColumn"] = {"values": values}
        all_values.append(values)

    curve.add_data(props, property_group="myGroup")

    return curve, all_values


def test_create_property_group(tmp_path):
    #  pylint: disable=too-many-locals
    # pylint: disable=too-many-statements

    h5file_path = tmp_path / r"prop_group_test.geoh5"

    with Workspace.create(h5file_path) as workspace:
        curve, _ = make_example(workspace)

        props = [child for child in curve.children if isinstance(child, Data)]

        test_values = np.r_[[prop.values for prop in props]]

        # Property group object should have been created
        prop_group = curve.fetch_property_group(name="myGroup")

        assert [
            "Period1",
            "Period2",
            "Period3",
            "Period4",
        ] == prop_group.properties_name

        prop_group.allow_delete = False

        assert prop_group.allow_delete is False

        assert prop_group.parent == curve

        single_data_group = curve.add_data_to_group(props[1], "Singleton")

        assert (
            len(single_data_group.properties) == 1  # 2
        ), "Failed adding data to property group."

        # Add data to group by uid
        single_data_group.add_properties(props[2].uid)

        assert (
            len(single_data_group.properties) == 2  # 3
        ), "Failed adding data to property group."

        # Remove data from group by data
        single_data_group.remove_properties(props[2])
        assert len(single_data_group.properties) == 1  # 2

        # get property group
        property_group_test = workspace.get_entity("myGroup")[0]
        assert isinstance(property_group_test, PropertyGroup)

        assert isinstance(workspace.property_groups, list)

        property_group_from_object = curve.get_entity("myGroup")[0]

        assert property_group_from_object == property_group_test

        np.testing.assert_almost_equal(
            np.r_[property_group_from_object.collect_values], np.array(test_values)
        )

    # Re-open the workspace
    with Workspace(h5file_path) as workspace:
        # problem with this line with some ubuntu and macos versions
        # assert workspace.get_entity("myGroup")[0].uid == property_group_test.uid

        rec_object = workspace.get_entity(curve.uid)[0]
        # Read the property_group back in
        rec_prop_group = rec_object.fetch_property_group(name="myGroup")

        attrs = rec_prop_group.attribute_map
        check_list = [
            attr
            for attr in attrs.values()
            if getattr(rec_prop_group, attr) != getattr(prop_group, attr)
        ]

        assert len(check_list) == 0, (
            f"Attribute{check_list} of PropertyGroups in output differ from input"
        )

        # Copy an object without children
        new_curve = rec_object.copy(copy_children=False)

        assert new_curve.property_groups is None, (
            "Property_groups not properly removed on copy without children."
        )

        #
        rec_object.remove_children(rec_prop_group)
        assert len(rec_object.property_groups) == 1, "Failed to remove property group"

        rec_object.remove_children(rec_object.property_groups[0])

    with Workspace(h5file_path) as workspace:
        rec_object = workspace.get_entity(curve.uid)[0]
        assert rec_object.property_groups is None, (
            "Property_groups not properly removed."
        )

        # test for properties post assignation
        properties = [child for child in rec_object.children if isinstance(child, Data)]
        prop_group = PropertyGroup(
            parent=rec_object, name="testGroup", properties=properties
        )

        assert [
            child.uid for child in rec_object.children if isinstance(child, Data)
        ] == prop_group.properties

        # empty property group
        empty_group = PropertyGroup(
            parent=rec_object, name="emptyGroup", association="CELL"
        )

        assert empty_group.collect_values is None
        assert empty_group.properties_name is None
        assert empty_group.table() is None

        empty_group.remove_properties("test")  # nothing is done


def test_property_group_errors(tmp_path):
    #  pylint: disable=too-many-locals
    # pylint: disable=too-many-statements

    h5file_path = tmp_path / r"prop_group_test.geoh5"

    with Workspace.create(h5file_path) as workspace:
        curve, _ = make_example(workspace)

        prop_group = curve.fetch_property_group(name="myGroup")

        with pytest.raises(TypeError, match="Name must be"):
            PropertyGroup(parent=curve, name=42)  # type: ignore

        with pytest.raises(TypeError, match="Parent bidon"):
            PropertyGroup(parent="bidon")  # type: ignore

        with pytest.raises(TypeError, match="Data must be of type Data"):
            PropertyGroup(parent=curve, properties=[123])

        with pytest.raises(TypeError, match="Could not convert input uid"):
            prop_group.uid = 123

        with pytest.raises(TypeError, match="Attribute 'on_file' must be a boolean"):
            prop_group.on_file = "bidon"

        with pytest.raises(
            ValueError, match="At least one of 'properties' or 'association'"
        ):
            PropertyGroup(parent=curve)

        with pytest.raises(TypeError, match="Property group must be of type"):
            curve.add_data_to_group(data="bidon", property_group=123)

        # test error for allow delete
        with pytest.raises(TypeError, match="allow_delete must be a boolean"):
            prop_group.allow_delete = "bidon"

        with pytest.raises(TypeError, match="Association must be"):
            PropertyGroup(parent=curve, association=123)

        with pytest.raises(TypeError, match="'Property group type' must be of type"):
            PropertyGroup(parent=curve, property_group_type=123)

        with pytest.raises(ValueError, match="'Property group type' must be one of"):
            PropertyGroup(parent=curve, property_group_type="badType")

        with pytest.raises(ValueError, match="Data 'bidon' not found"):
            prop_group._validate_data("bidon")  # pylint: disable=protected-access

        curve.add_data(
            {"Period1": {"values": np.random.rand(12)}}, property_group="myGroup"
        )

        curve.add_data(
            {"TestAssociation": {"values": np.random.rand(11), "association": "CELL"}},
        )

        with pytest.raises(ValueError, match="Data 'TestAssociation' association"):
            prop_group._validate_data("TestAssociation")  # pylint: disable=protected-access

        test = Curve.create(
            workspace,
            vertices=np.c_[np.linspace(0, 2 * np.pi, 11), np.zeros(11), np.zeros(11)],
        )

        test_data = test.add_data(
            {"WrongParent": {"values": np.random.rand(10), "association": "CELL"}},
        )

        with pytest.raises(ValueError, match="Data 'WrongParent' parent"):
            prop_group._validate_data(test_data)  # pylint: disable=protected-access


def test_auto_find_association(tmp_path):
    h5file_path = tmp_path / r"prop_group_test.geoh5"

    with Workspace.create(h5file_path) as workspace:
        curve, _ = make_example(workspace)

        prop_group = PropertyGroup(
            parent=curve,
            name="myGroup2",
            properties=[data for data in curve.children if isinstance(data, Data)],
        )

        assert prop_group.association == DataAssociationEnum.VERTEX


def test_copy_property_group(tmp_path):
    h5file_path = tmp_path / r"prop_group_test.geoh5"

    with Workspace.create(h5file_path) as workspace:
        curve, _ = make_example(workspace)

        # New property group object should have been created on copy
        curve_2 = curve.copy()

        assert len(curve_2.property_groups) > 0
        assert all(
            len(curve_2.get_data(uid)) > 0
            for uid in curve_2.property_groups[0].properties
        )


def test_property_group_same_name(tmp_path):
    h5file_path = tmp_path / f"{__name__}.geoh5"

    with Workspace.create(h5file_path) as workspace:
        curve, _ = make_example(workspace)

        pg2 = PropertyGroup(parent=curve, name="myGroup", association="VERTEX")

        assert pg2.name == "myGroup(1)"

    with Workspace(h5file_path) as workspace:
        # error here if a property group has the same name
        curve = workspace.get_entity(curve.uid)[0]

        assert sorted([pg.name for pg in curve.property_groups]) == [
            "myGroup",
            "myGroup(1)",
        ]


def test_clean_out_empty(tmp_path):
    h5file_path = tmp_path / r"prop_group_clean.geoh5"

    with Workspace.create(h5file_path) as workspace:
        curve, _ = make_example(workspace)

        assert len(curve.property_groups) == 1

        props = [child for child in curve.children if isinstance(child, Data)]
        curve.remove_children(props)

        assert len(curve.property_groups) == 0


def test_property_group_table(tmp_path):
    h5file_path = tmp_path / r"prop_group_test.geoh5"

    with Workspace.create(h5file_path) as workspace:
        curve, expected = make_example(workspace, add_str_column=True)
        expected = np.array(expected, dtype="O").T

        # Property group object should have been created
        prop_group = curve.fetch_property_group(name="myGroup")

        prop_table = prop_group.table(spatial_index=True)
        produced = np.array([tuple(row) for row in prop_table], dtype="O")

        np.testing.assert_almost_equal(expected[:, :-1], produced[:, 3:-1])

        assert all(expected[:, -1] == produced[:, -1])

        np.testing.assert_almost_equal(curve.locations, produced[:, :3], decimal=6)

        curve.add_data(
            {
                "TestCell": {"values": np.random.rand(11), "association": "CELL"},
                "Referenced": {
                    "values": np.random.randint(0, 3, 11),
                    "association": "CELL",
                    "type": "referenced",
                    "value_map": {1: "A", 2: "B", 3: "C"},
                },
            },
            property_group="cellGroup",
        )

        # Property group object should have been created
        cell_group = curve.fetch_property_group(name="cellGroup")

        np.testing.assert_almost_equal(cell_group.table.locations, curve.centroids)

        assert cell_group.table.size == curve.n_cells

        assert isinstance(cell_group.table(mapped=True)[0][1], str)


def test_property_group_table_error(tmp_path):
    h5file_path = tmp_path / r"prop_group_test.geoh5"

    with Workspace.create(h5file_path) as workspace:
        curve = Curve.create(
            workspace,
            vertices=np.c_[np.linspace(0, 2 * np.pi, 12), np.zeros(12), np.zeros(12)],
        )

        curve.add_data(
            {"test": {"values": np.random.rand(12), "association": "UNKNOWN"}},
            property_group="myGroup",
        )

        prop_group = curve.fetch_property_group(name="myGroup")

        with pytest.raises(ValueError, match="The association DataAssociation"):
            _ = prop_group.table.locations

        with pytest.raises(ValueError, match="The association DataAssociation"):
            _ = prop_group.table.size

        with pytest.raises(TypeError, match="'property_group' must be a PropertyGroup"):
            PropertyGroupTable(property_group=123)  # type: ignore

        drillhole = Drillhole.create(workspace, name="test")

        prop_group = PropertyGroup(
            parent=drillhole, name="drillhole", association="DEPTH"
        )

        with pytest.raises(
            NotImplementedError, match="PropertyGroupTable is not supported"
        ):
            _ = prop_group.table

        strike_dip = curve.add_data(
            {
                "Strike": {"values": np.random.rand(12), "association": "VERTEX"},
                "Dip": {"values": np.random.rand(12), "association": "VERTEX"},
            },
            property_group="strikeDip",
        )

        prop_group = PropertyGroup(
            parent=curve,
            name="strikeDip",
            property_group_type="Strike & dip",
            properties=strike_dip,
        )

        with pytest.raises(ValueError, match="Cannot add properties to "):
            prop_group.add_properties(curve)

        with pytest.raises(ValueError, match="Cannot remove properties from "):
            prop_group.remove_properties(curve)


def test_group_type_enum(tmp_path):
    workspace = Workspace(tmp_path / "test.geoh5")
    curve = Curve.create(
        workspace,
        vertices=np.c_[np.linspace(0, 2 * np.pi, 12), np.zeros(12), np.zeros(12)],
    )
    data = curve.add_data(
        {"test": {"values": np.random.rand(12), "association": "VERTEX"}},
        property_group="myGroup",
    )

    data_text = curve.add_data(
        {
            "text": {
                "values": np.array(["i" for i in range(12)]),
                "association": "VERTEX",
            }
        },
        property_group="myGroup2",
    )

    with pytest.raises(TypeError, match="First children of 'Depth table'"):
        GroupTypeEnum("Depth table").verify([data])

    with pytest.raises(TypeError, match="Children of 'Dip direction & dip'"):
        GroupTypeEnum("Dip direction & dip").verify([data])

    with pytest.raises(TypeError, match="First two children of 'Interval table'"):
        GroupTypeEnum("Interval table").verify([data])

    with pytest.raises(TypeError, match="Children of 'Multi-element'"):
        GroupTypeEnum("Multi-element").verify([data_text])

    with pytest.raises(TypeError, match="Children of 'Strike & dip'"):
        GroupTypeEnum("Strike & dip").verify([data])

    with pytest.raises(TypeError, match="Children of '3D vector'"):
        GroupTypeEnum("3D vector").verify([data])
