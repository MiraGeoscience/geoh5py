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

# pylint: disable=R0914
# mypy: ignore-errors

from __future__ import annotations

import numpy as np
import pytest

from geoh5py.data import FloatData, data_type
from geoh5py.groups import ContainerGroup, DrillholeGroup, Group
from geoh5py.objects import Drillhole, ObjectBase
from geoh5py.shared import fetch_h5_handle
from geoh5py.shared.concatenation import (
    ConcatenatedData,
    ConcatenatedObject,
    ConcatenatedPropertyGroup,
    Concatenator,
)
from geoh5py.shared.utils import as_str_if_uuid, compare_entities
from geoh5py.workspace import Workspace


def test_concatenator(tmp_path):
    h5file_path = tmp_path / r"test_Concatenator.geoh5"

    with Workspace.create(h5file_path, version=2.0) as workspace:
        # Create a workspace
        dh_group = DrillholeGroup.create(workspace)

        assert (
            dh_group.data == {}
        ), "DrillholeGroup should not have data on instantiation."

        with pytest.raises(ValueError) as error:
            dh_group.concatenated_attributes = "123"

        assert "Input 'concatenated_attributes' must be a dictionary or None" in str(
            error
        )

        with pytest.raises(AttributeError) as error:
            dh_group.concatenated_object_ids = "123"

        assert "Input value for 'concatenated_object_ids' must be of type list." in str(
            error
        )

        with pytest.raises(ValueError) as error:
            dh_group.data = "123"

        assert "Input 'data' must be a dictionary" in str(error)

        with pytest.raises(ValueError) as error:
            dh_group.index = "123"

        assert "Input 'index' must be a dictionary" in str(error)

        assert dh_group.fetch_concatenated_objects() == {}

        dh_group_copy = dh_group.copy()

        compare_entities(dh_group_copy, dh_group, ignore=["_uid"])


def test_concatenated_entities(tmp_path):
    h5file_path = tmp_path / r"test_concatenated_data.geoh5"
    with Workspace.create(h5file_path, version=2.0) as workspace:
        class_type = type("TestGroup", (Concatenator, ContainerGroup), {})
        entity_type = Group.find_or_create_type(workspace)
        concat = class_type(entity_type)

        class_obj_type = type("TestObject", (ConcatenatedObject, Drillhole), {})
        object_type = ObjectBase.find_or_create_type(workspace)

        with pytest.raises(UserWarning) as error:
            concat_object = class_obj_type(object_type)

        assert (
            "Creating a concatenated object must have a parent of type Concatenator."
            in str(error)
        )

        concat_object = class_obj_type(object_type, parent=concat)

        with pytest.raises(UserWarning) as error:
            class_type = type("TestData", (ConcatenatedData, FloatData), {})
            dtype = data_type.DataType.find_or_create(
                workspace, primitive_type=FloatData.primitive_type()
            )
            data = class_type(dtype)

        assert "Creating a concatenated data must have a parent" in str(error)

        data = class_type(dtype, parent=concat_object)

        assert data.property_group is None

        with pytest.raises(UserWarning) as error:
            prop_group = ConcatenatedPropertyGroup(None)

        assert "Creating a concatenated data must have a parent" in str(error)

        prop_group = ConcatenatedPropertyGroup(parent=concat_object)

        with pytest.raises(
            AttributeError, match="Cannot change parent of a property group."
        ):
            prop_group.parent = Drillhole

        assert prop_group.to_ is None
        assert prop_group.from_ is None

        setattr(prop_group, "_parent", None)

        with pytest.raises(
            AttributeError, match="The 'parent' of a concatenated Data must be of type"
        ):
            prop_group.parent = "bidon"

        prop_group.parent = concat_object

        assert prop_group.parent == concat_object

        with pytest.raises(KeyError, match="A Property Group"):
            concat_object.create_property_group(name="property_group")


def test_empty_concatenated_property_group():
    workspace = Workspace()
    dh_group = DrillholeGroup.create(workspace)
    well = Drillhole.create(
        workspace,
        parent=dh_group,
    )
    ConcatenatedPropertyGroup(parent=well)
    assert not well.from_


def test_create_drillhole_data(tmp_path):  # pylint: disable=too-many-statements
    h5file_path = tmp_path / r"test_drillholeGroup.geoh5"
    new_path = tmp_path / r"test_drillholeGroup2.geoh5"
    well_name = "bullseye/"
    n_data = 10

    with Workspace.create(h5file_path, version=2.0) as workspace:
        # Create a workspace
        dh_group = DrillholeGroup.create(workspace)

        well = Drillhole.create(
            workspace,
            collar=np.r_[0.0, 10.0, 10],
            surveys=np.c_[
                np.linspace(0, 100, n_data),
                np.ones(n_data) * 45.0,
                np.linspace(-89, -75, n_data),
            ],
            parent=dh_group,
            name=well_name,
        )

        # Plain drillhole
        singleton = Drillhole.create(
            workspace,
        )
        with pytest.warns(match="Expected a Concatenated object"):
            singleton.parent = dh_group

        assert len(dh_group.children) == 1

        dh_group.add_comment("This is a comment")

        assert len(dh_group.children) == 2

        with pytest.raises(UserWarning, match="does not have a property or values"):
            dh_group.update_array_attribute(well, "abc")

        # Add both set of log data with 0.5 m tolerance
        values = np.random.randn(50)
        with pytest.raises(
            UserWarning, match="Input depth 'collocation_distance' must be >0."
        ):
            well.add_data(
                {
                    "my_log_values/": {
                        "depth": np.arange(0, 50.0),
                        "values": values,
                    }
                },
                collocation_distance=-1.0,
            )

        # Add both set of log data with 0.5 m tolerance
        with pytest.raises(AttributeError, match="Input data dictionary must contain"):
            well.add_data(
                {
                    "my_log_values/": {
                        "values": np.random.randn(50),
                    }
                },
            )

        test_values = np.random.randn(30)
        test_values[0] = np.nan
        test_values[-1] = np.nan

        well.add_data(
            {
                "my_log_values/": {
                    "depth": np.arange(0, 50.0),
                    "values": np.random.randn(50),
                },
                "log_wt_tolerance": {
                    "depth": np.arange(0.01, 50.01),
                    "values": test_values.astype(np.float32),
                },
            }
        )

        log_data = well.get_data("my_log_values/")[0]
        assert log_data.parent.name == "bullseye/"
        assert (
            log_data.parent.depth_
            and not log_data.parent.from_
            and not log_data.parent.to_
        )
        assert log_data.parent.depth_[0].name == "DEPTH"
        assert not log_data.parent.get_data("FROM")
        assert not log_data.parent.get_data("TO")
        assert log_data.property_group.name == "depth_0"
        assert log_data.property_group.depth_.name == "DEPTH"
        assert (
            len(log_data.parent.property_groups[0].properties) == 3
        ), "Should only have 3 properties (my_log_values, log_wt_tolerance, DEPTH)"
        assert log_data.property_group.property_group_type == "Depth table"

        with fetch_h5_handle(h5file_path) as h5file:
            name = list(h5file)[0]
            group = h5file[name]["Groups"][as_str_if_uuid(dh_group.uid)][
                "Concatenated Data"
            ]
            assert np.all(~np.isnan(group["Data"]["log_wt_tolerance"][:]))

        assert len(well.get_data("my_log_values/")) == 1
        assert len(well.get_data("my_log_values/")[0].values) == 50

        with pytest.raises(UserWarning, match="already present on the drillhole"):
            well.add_data(
                {
                    "my_log_values/": {
                        "depth": np.arange(0, 50.0),
                        "values": np.random.randn(50),
                    },
                }
            )

        well_b = well.copy()
        well_b.name = "Number 2"
        well_b.collar = np.r_[10.0, 10.0, 10]

        # Create random from-to
        from_to_a = np.sort(np.random.uniform(low=0.05, high=100, size=(50,))).reshape(
            (-1, 2)
        )
        from_to_b = np.vstack([from_to_a[0, :], [30.1, 55.5], [56.5, 80.2]])

        # Add from-to data
        data_objects = well.add_data(
            {
                "interval_values": {
                    "values": np.random.randn(from_to_a.shape[0]),
                    "from-to": from_to_a.tolist(),
                },
                "int_interval_list": {
                    "values": np.asarray([1, 2, 3]),
                    "from-to": from_to_b.T.tolist(),
                    "value_map": {1: "Unit_A", 2: "Unit_B", 3: "Unit_C"},
                    "type": "referenced",
                },
                "interval_values_b": {
                    "values": np.random.randn(from_to_b.shape[0]),
                    "from-to": from_to_b,
                },
            }
        )

        interval_data = data_objects[0]
        assert interval_data.property_group.name == "Interval_0"
        assert interval_data.parent.get_data("FROM")

        assert interval_data.parent.get_data("FROM(1)")
        assert interval_data.parent.get_data("TO")
        assert interval_data.parent.get_data("TO(1)")
        assert (
            (interval_data.property_group.depth_ is None)
            and (interval_data.property_group.from_ is not None)
            and (interval_data.property_group.to_ is not None)
        )
        assert interval_data.property_group.property_group_type == "Interval table"
        assert data_objects[1].property_group == data_objects[2].property_group

        well_b_data = well_b.add_data(
            {
                "interval_values": {
                    "values": np.random.randn(from_to_b.shape[0]),
                    "from-to": from_to_b,
                },
            }
        )
        depth_data = well_b.add_data(
            {
                "Depth Data": {
                    "depth": np.sort(np.random.uniform(low=0.05, high=100, size=(10,))),
                    "values": np.random.randn(8),
                },
            }
        )

        assert dh_group.fetch_index(well_b_data, well_b_data.name) == 1, (
            "'interval_values' on well_b should be the second entry.",
        )

        assert len(well.to_) == len(well.from_) == 2, "Should have only 2 from-to data."

        with pytest.raises(
            UserWarning, match="Data with name 'Depth Data' already present"
        ):
            well_b.add_data(
                {
                    "Depth Data": {
                        "values": np.random.randn(10),
                        "depth": np.sort(
                            np.random.uniform(low=0.05, high=100, size=(10,))
                        ),
                    },
                }
            )

        well_b_data.values = np.random.randn(from_to_b.shape[0])

    with well.workspace.open(mode="r") as workspace:
        # Check entities
        compare_entities(
            well,
            workspace.get_entity(well_name)[0],
            ignore=[
                "_parent",
                "_metadata",
                "_default_collocation_distance",
                "_property_groups",
            ],
        )

        compare_entities(
            data_objects[0],
            well.get_data("interval_values")[0],
            ignore=["_metadata", "_parent"],
        )
        compare_entities(
            data_objects[1],
            well.get_entity("int_interval_list")[0],
            ignore=["_metadata", "_parent"],
        )
        compare_entities(
            data_objects[2],
            well.get_entity("interval_values_b")[0],
            ignore=["_metadata", "_parent"],
        )

        well_b_reload = workspace.get_entity("Number 2")[0]

        compare_entities(
            well_b,
            well_b_reload,
            ignore=[
                "_parent",
                "_metadata",
                "_default_collocation_distance",
                "_property_groups",
                "_uid",
            ],
        )
        compare_entities(
            depth_data,
            well_b_reload.get_data("Depth Data")[0],
            ignore=["_metadata", "_parent"],
        )

        assert (
            workspace.get_entity("Number 2")[0].get_data_list()
            == well_b.get_data_list()
        )

        with Workspace.create(new_path, version=2.0) as new_workspace:
            new_group = dh_group.copy(parent=new_workspace)
            well = [k for k in new_group.children if k.name == "bullseye/"][0]

            prop_group = [k for k in well.property_groups if k.name == "Interval_0"][0]
            with pytest.raises(
                ValueError, match="Input values for 'new_data' with shape"
            ):
                well.add_data(
                    {
                        "new_data": {"values": np.random.randn(24).astype(np.float32)},
                    },
                    property_group=prop_group.name,
                )

            well.add_data(
                {
                    "new_data": {"values": np.random.randn(25).astype(np.float32)},
                },
                property_group=prop_group.name,
            )

        assert (
            len(well.property_groups[0].properties) == 3
        ), "Issue adding data to interval."


def create_drillholes(h5file_path, version=1.0, ga_version="1.0"):
    well_name = "well"
    n_data = 10

    with Workspace.create(
        h5file_path, version=version, ga_version=ga_version
    ) as workspace:
        # Create a workspace
        dh_group = DrillholeGroup.create(workspace, name="DH_group")
        well = Drillhole.create(
            workspace,
            collar=np.r_[0.0, 10.0, 10],
            surveys=np.c_[
                np.linspace(0, 100, n_data),
                np.ones(n_data) * 45.0,
                np.linspace(-89, -75, n_data),
            ],
            parent=dh_group,
            name=well_name,
        )
        # Create random from-to
        from_to_a = np.sort(np.random.uniform(low=0.05, high=100, size=(50,))).reshape(
            (-1, 2)
        )
        from_to_b = np.vstack([from_to_a[0, :], [30.1, 55.5], [56.5, 80.2]])

        values = np.random.randn(50)
        values[0] = np.nan
        # Add both set of log data with 0.5 m tolerance
        well.add_data(
            {
                "my_log_values/": {
                    "depth": np.arange(0, 50.0),
                    "values": np.random.randn(50),
                },
                "log_wt_tolerance": {
                    "depth": np.arange(0.01, 50.01),
                    "values": values,
                },
            }
        )
        well_b = well.copy()
        well_b.name = "Number 2"
        well_b.collar = np.r_[10.0, 10.0, 10]

        well_c = well.copy()
        well_c.name = "Number 3"
        well_c.collar = np.r_[10.0, -10.0, 10]

        well.add_data(
            {
                "interval_values_b": {
                    "values": np.random.randn(from_to_b.shape[0]),
                    "from-to": from_to_b,
                },
            }
        )
    return dh_group, workspace


def test_remove_drillhole_data(tmp_path):
    h5file_path = tmp_path / r"test_remove_concatenated.geoh5"

    create_drillholes(h5file_path, version=2.0, ga_version="1.0")

    with Workspace(h5file_path, version=2.0) as workspace:
        well = workspace.get_entity("well")[0]
        well_b = workspace.get_entity("Number 2")[0]
        data = well.get_data("my_log_values/")[0]
        workspace.remove_entity(data)
        workspace.remove_entity(well_b)

        assert np.isnan(well.get_data("log_wt_tolerance")[0].values).sum() == 1

    with Workspace(h5file_path, version=2.0) as workspace:
        well = workspace.get_entity("well")[0]
        assert "my_log_values/" not in well.get_entity_list()
        assert workspace.get_entity("Number 2")[0] is None


def test_create_drillhole_data_v4_2(tmp_path):
    h5file_path = tmp_path / r"test_create_concatenated_v4_2_v2_1.geoh5"
    dh_group, workspace = create_drillholes(h5file_path, version=2.1, ga_version="4.2")

    with workspace.open():
        assert dh_group.workspace.ga_version == "4.2"
        assert dh_group.concat_attr_str == "Attributes Jsons"
        assert (
            len(workspace.fetch_children(dh_group, recursively=True)) == 3
        ), "Issue with fetching children recursively"

    h5file_path = tmp_path / r"test_create_concatenated_v4_2_v2_0.geoh5"
    dh_group, workspace = create_drillholes(h5file_path, version=2.0, ga_version="4.2")
    with workspace.open():
        assert dh_group.workspace.ga_version == "4.2"
        assert dh_group.workspace.version == 2.0
        assert dh_group.concat_attr_str == "Attributes"


def test_copy_drillhole_group(tmp_path):
    h5file_path = tmp_path / r"test_copy_concatenated.geoh5"

    _, workspace = create_drillholes(h5file_path, version=2.0, ga_version="4.2")

    with workspace.open():
        dh_group = workspace.get_entity("DH_group")[0]
        dh_group_copy = dh_group.copy(workspace)

        for child_a, child_b in zip(dh_group.children, dh_group_copy.children):
            assert child_a.name == child_b.name
            assert child_a.collar == child_b.collar
            np.testing.assert_array_almost_equal(child_a.surveys, child_b.surveys)
            assert child_a.get_data_list() == child_b.get_data_list()

        with Workspace.create(
            tmp_path / r"test_copy_concatenated_copy.geoh5",
            version=2.0,
            ga_version="4.2",
        ) as new_space:
            dh_group_copy = dh_group.copy(parent=new_space)
            compare_entities(
                dh_group_copy,
                dh_group,
                ignore=["_metadata", "_parent", "_data", "_index"],
            )


def test_copy_from_extent_drillhole_group(tmp_path):
    h5file_path = tmp_path / r"test_copy_concatenated.geoh5"

    _, workspace = create_drillholes(h5file_path, version=2.0, ga_version="4.2")

    with workspace.open():
        dh_group = workspace.get_entity("DH_group")[0]
        dh_group_copy = dh_group.copy_from_extent(extent=np.asarray([[0, 0], [15, 15]]))

        assert len(dh_group_copy.children) == 2
        for child_a in dh_group_copy.children:
            child_b = dh_group.get_entity(child_a.name)[0]
            assert child_a.name == child_b.name
            assert child_a.collar == child_b.collar
            np.testing.assert_array_almost_equal(child_a.surveys, child_b.surveys)
            assert child_a.get_data_list() == child_b.get_data_list()


def test_open_close_creation(tmp_path):
    h5file_path = tmp_path / r"test_drillholeGroup.geoh5"

    with Workspace.create(h5file_path, version=2.0) as workspace:
        # Create a workspace
        dh_group = DrillholeGroup.create(workspace)

        Drillhole.create(
            workspace,
            parent=dh_group,
            name="DH1",
        )

    workspace.open()
    Drillhole.create(
        workspace,
        parent=dh_group,
        name="DH2",
    )
    assert len(workspace.groups[1].concatenated_attributes["Attributes"]) == 2
    workspace.close()
