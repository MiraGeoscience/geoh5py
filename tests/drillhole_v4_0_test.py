# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2025-2026 Mira Geoscience Ltd.                                '
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


# pylint: disable=too-many-lines
# pylint: disable=too-many-locals
# mypy: ignore-errors

from __future__ import annotations

import random
import string

import numpy as np
import pytest
from h5py import special_dtype

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
from geoh5py.shared.concatenation.drillholes_group_table import DrillholesGroupTable
from geoh5py.shared.utils import as_str_if_uuid, compare_entities
from geoh5py.workspace import Workspace


def create_drillholes(h5file_path, version=1.0, ga_version="1.0", add_data=True):
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
        well_b = well.copy()
        well_b.name = "Number 2"
        well_b.collar = np.r_[10.0, 10.0, 10]

        well_c = well.copy()
        well_c.name = "Number 3"
        well_c.collar = np.r_[10.0, -10.0, 10]

        if add_data:
            # Create random from-to
            from_to_a = np.sort(
                np.random.uniform(low=0.05, high=100, size=(50,))
            ).reshape((-1, 2))

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

            well.add_data(
                {
                    "text Data": {
                        "values": np.array(
                            [
                                "".join(
                                    random.choice(string.ascii_lowercase)
                                    for _ in range(6)
                                )
                                for _ in range(from_to_a.shape[0])
                            ]
                        ),
                        "from-to": from_to_a,
                        "type": "TEXT",
                    },
                    "interval_values_a": {
                        "values": np.random.randn(from_to_a.shape[0]),
                        "from-to": from_to_a,
                    },
                },
                property_group="property_group",
            )
            well_c.add_data(
                {
                    "interval_values_b": {
                        "values": np.random.randn(from_to_a.shape[0]),
                        "from-to": from_to_a,
                    },
                },
                property_group="property_group",
            )

    return dh_group, workspace


def test_concatenator(tmp_path):
    h5file_path = tmp_path / r"test_Concatenator.geoh5"

    xyz = np.random.randn(32)
    np.savetxt(tmp_path / r"numpy_array.txt", xyz)
    file_name = "numpy_array.txt"

    with Workspace.create(h5file_path, version=2.0) as workspace:
        # Create a workspace
        dh_group = DrillholeGroup.create(workspace, name="DH_group")
        dh_group.add_file(tmp_path / file_name)
        assert dh_group.data == {}, (
            "DrillholeGroup should not have data on instantiation."
        )

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

    with Workspace(h5file_path) as workspace:
        assert len(workspace.get_entity("DH_group")[0].children) == 1


def test_concatenated_entities(tmp_path):
    h5file_path = tmp_path / r"test_concatenated_data.geoh5"
    with Workspace.create(h5file_path, version=2.0) as workspace:
        class_type = type("TestGroup", (Concatenator, ContainerGroup), {})
        entity_type = Group.find_or_create_type(workspace)
        concat = class_type(entity_type=entity_type)

        class_obj_type = type("TestObject", (ConcatenatedObject, Drillhole), {})
        object_type = ObjectBase.find_or_create_type(workspace)

        with pytest.raises(UserWarning) as error:
            concat_object = class_obj_type(entity_type=object_type)

        assert (
            "Creating a concatenated object must have a parent of type Concatenator."
            in str(error)
        )

        concat_object = class_obj_type(entity_type=object_type, parent=concat)

        with pytest.raises(UserWarning) as error:
            class_type = type("TestData", (ConcatenatedData, FloatData), {})
            dtype = data_type.DataType.find_or_create(
                workspace, primitive_type=FloatData
            )
            class_type(entity_type=dtype)

        assert "Creating a concatenated data must have a parent" in str(error)

        data = class_type(entity_type=dtype, parent=concat_object)

        assert data.property_group is None

        with pytest.raises(TypeError, match="must have a 'property_groups' attribute"):
            ConcatenatedPropertyGroup(None)

        prop_group = ConcatenatedPropertyGroup(parent=concat_object, properties=[data])

        with pytest.raises(
            AttributeError, match="Cannot change parent of a property group."
        ):
            prop_group.parent = Drillhole

        assert prop_group.to_ is None
        assert prop_group.from_ is None

        prop_group._parent = None

        with pytest.raises(
            ValueError,
            match="The 'parent' of a concatenated data must have an 'add_children' method.",
        ):
            prop_group.parent = "bidon"

        prop_group.parent = concat_object

        assert prop_group.parent == concat_object


def test_empty_concatenated_property_group():
    workspace = Workspace()
    dh_group = DrillholeGroup.create(workspace)
    well = Drillhole.create(
        workspace,
        parent=dh_group,
    )
    ConcatenatedPropertyGroup(parent=well, association="DEPTH")
    assert not well.from_


def test_create_drillhole_data(tmp_path):  # pylint: disable=too-many-statements
    h5file_path = tmp_path / r"test_drillholeGroup.geoh5"
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
        values = np.random.randn(48)
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

        with pytest.raises(ValueError, match="Mismatch between input"):
            well.add_data(
                {
                    "my_log_values/": {
                        "depth": np.arange(0, 49.0).tolist(),
                        "values": np.random.randn(50),
                    },
                }
            )

        test_values = np.random.randn(30)
        test_values[0] = np.nan
        test_values[-1] = np.nan

        well.add_data(
            {
                "my_log_values/": {
                    "depth": np.arange(0, 50.0).tolist(),
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
        assert len(log_data.parent.property_groups[0].properties) == 3, (
            "Should only have 3 properties (my_log_values, log_wt_tolerance, DEPTH)"
        )
        assert log_data.property_group.property_group_type == "Depth table"

        with fetch_h5_handle(h5file_path) as h5file:
            name = list(h5file)[0]
            group = h5file[name]["Groups"][as_str_if_uuid(dh_group.uid)][
                "Concatenated Data"
            ]
            assert np.all(~np.isnan(group["Data"]["log_wt_tolerance"][:]))

        assert len(well.get_data("my_log_values/")) == 1
        assert len(well.get_data("my_log_values/")[0].values) == 50


def test_append_data_to_tables(tmp_path):
    h5file_path = tmp_path / r"test_append_data_to_tables.geoh5"

    _, workspace = create_drillholes(h5file_path, version=2.0, ga_version="1.0")

    with workspace.open():
        well = workspace.get_entity("well")[0]
        dh_group = well.parent
        well_b = workspace.get_entity("Number 2")[0]
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
                    "values": np.random.randn(from_to_a.shape[0] - 1),
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
        assert interval_data.property_group.name == "Interval_1"
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
        text = np.array(
            [
                "".join(random.choice(string.ascii_lowercase) for _ in range(6))
                for _ in range(from_to_b.shape[0])
            ]
        )
        text_data = well_b.add_data(
            {
                "text Data": {
                    "values": text,
                    "from-to": from_to_b,
                    "type": "TEXT",
                },
            }
        )

        assert dh_group.fetch_index(well_b_data, well_b_data.name) == 1, (
            "'interval_values' on well_b should be the second entry.",
        )

        assert len(well.to_) == len(well.from_) == 3, (
            "Should have exactly 3 from-to data."
        )

        well_b_data.values = np.random.randn(from_to_b.shape[0])

    with well.workspace.open(mode="r") as workspace:
        # Check entities
        compare_entities(
            well,
            workspace.get_entity("well")[0],
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
            decimal=5,
        )

        compare_entities(
            depth_data,
            well_b_reload.get_data("Depth Data")[0],
            ignore=["_metadata", "_parent"],
        )

        compare_entities(
            text_data,
            well_b_reload.get_data("text Data")[0],
            ignore=["_metadata", "_parent"],
        )

        assert (
            workspace.get_entity("Number 2")[0].get_data_list()
            == well_b.get_data_list()
        )


def test_copy_and_append_drillhole_data(tmp_path):
    h5file_path = tmp_path / r"test_copy_and_append_drillhole_data.geoh5"

    _, workspace = create_drillholes(h5file_path, version=2.0, ga_version="1.0")

    new_path = tmp_path / r"test_copy_and_append_drillhole_data_NEW.geoh5"

    with workspace.open():
        dh_group = workspace.get_entity("DH_group")[0]
        with Workspace.create(new_path, version=2.0) as new_workspace:
            new_group = dh_group.copy(parent=new_workspace)
            well = [k for k in new_group.children if k.name == "well"][0]

            prop_group = [
                k for k in well.property_groups if k.name == "property_group"
            ][0]
            with pytest.raises(ValueError, match="Input values with shape"):
                well.add_data(
                    {
                        "new_data": {"values": np.random.randn(24).astype(np.float32)},
                    },
                    property_group=prop_group.name,
                )

            well.add_data(
                {
                    "new_data": {
                        "values": np.random.randn(
                            prop_group.from_.values.shape[0]
                        ).astype(np.float32)
                    },
                },
                property_group=prop_group.name,
            )

            prop_group = [k for k in well.property_groups if k.name == "depth_0"][0]
            well.add_data(
                {
                    "new_data_depth": {
                        "values": np.random.randn(25).astype(np.float32)
                    },
                },
                property_group=prop_group.name,
            )

            with pytest.raises(AttributeError, match="Input data property group"):
                well.add_data(
                    {
                        "new_data_bidon": {
                            "values": np.random.randn(24).astype(np.float32)
                        },
                    },
                    property_group=ConcatenatedPropertyGroup(
                        parent=well,
                        property_group_type="Multi-element",
                        association="DEPTH",
                    ),
                )

            assert len(well.property_groups[0].properties) == 4, (
                "Issue adding data to interval."
            )


def test_partial_group_removal(tmp_path):
    h5file_path = tmp_path / r"test_append_data_to_tables.geoh5"

    _, workspace = create_drillholes(h5file_path, version=2.0, ga_version="1.0")

    with workspace.open():
        well = workspace.get_entity("well")[0]
        # Create random from-to
        from_to_a = np.sort(np.random.uniform(low=0.05, high=100, size=(50,))).reshape(
            (-1, 2)
        )
        from_to_b = np.vstack([from_to_a[0, :], [30.1, 55.5], [56.5, 80.2]])

        # Add from-to data
        well.add_data(
            {
                "interval_values": {
                    "values": np.random.randn(from_to_a.shape[0] - 1),
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

        well.remove_children(well.get_entity("interval_values_a"))
        well.remove_children(well.get_entity("text Data"))

    with workspace.open():
        well = workspace.get_entity("well")[0]
        assert len(well.property_groups) == 3


def test_remove_drillhole_data(tmp_path):
    h5file_path = tmp_path / r"test_remove_concatenated.geoh5"

    create_drillholes(h5file_path, version=2.0, ga_version="1.0")

    with Workspace(h5file_path, version=2.0) as workspace:
        well = workspace.get_entity("well")[0]

        assert np.isnan(well.get_data("log_wt_tolerance")[0].values).sum() == 1

        dh_group = well.parent

        data = well.get_data("my_log_values/")[0]
        new_well = well.copy(parent=dh_group, name="well copy")
        well.remove_children(data)
        del data
        assert "my_log_values/" not in well.get_entity_list()
        assert len(well.property_groups[0].properties) == 2

        assert workspace.get_entity("my_log_values/")[0] is not None
        dh_group.remove_children(new_well)

    with Workspace(h5file_path, version=2.0) as workspace:
        well = workspace.get_entity("well")[0]
        assert "my_log_values/" not in well.get_entity_list()
        assert workspace.get_entity("well copy")[0] is None

    with Workspace(h5file_path, version=2.0) as workspace:
        well = workspace.get_entity("well")[0]
        all_data = [well.get_entity(name)[0] for name in well.get_data_list()]
        all_data = [data for data in all_data if data.allow_delete]

        well.remove_children(all_data)

        assert len(well.property_groups) == 0

        well = workspace.get_entity("Number 3")[0]
        well.remove_children(well.property_groups[0])

        assert len(well.children) == 0, "Prop group and all data should be removed."


def test_create_drillhole_data_v4_2(tmp_path):
    h5file_path = tmp_path / r"test_create_concatenated_v4_2_v2_1.geoh5"
    dh_group, workspace = create_drillholes(h5file_path, version=2.1, ga_version="4.2")

    with workspace.open():
        assert dh_group.workspace.ga_version == "4.2"
        assert dh_group.concat_attr_str == "Attributes Jsons"
        assert len(workspace.fetch_children(dh_group, recursively=True)) == 3, (
            "Issue with fetching children recursively"
        )

    h5file_path = tmp_path / r"test_create_concatenated_v4_2_v2_0.geoh5"
    dh_group, workspace = create_drillholes(h5file_path, version=2.0, ga_version="4.2")
    with workspace.open():
        assert dh_group.workspace.ga_version == "4.2"
        assert dh_group.workspace.version == 2.0
        assert dh_group.concat_attr_str == "Attributes"


def test_copy_drillhole_group(tmp_path):
    h5file_path = tmp_path / r"test_copy_concatenated.geoh5"

    _, workspace = create_drillholes(h5file_path, version=2.0, ga_version="4.2")

    xyz = np.random.randn(32)
    np.savetxt(tmp_path / r"numpy_array.txt", xyz)
    file_name = "numpy_array.txt"

    with workspace.open():
        dh_group = workspace.get_entity("DH_group")[0]
        dh_group.add_file(tmp_path / file_name)
        dh_group_copy = dh_group.copy(workspace)

        for child_a, child_b in zip(
            dh_group.children, dh_group_copy.children, strict=False
        ):
            if isinstance(child_a, Drillhole):
                assert child_a.name == child_b.name
                assert child_a.collar == child_b.collar
                np.testing.assert_array_almost_equal(child_a.surveys, child_b.surveys)
                assert child_a.get_data_list() == child_b.get_data_list()
            else:
                assert child_a.values == child_b.values

        with Workspace.create(
            tmp_path / r"test_copy_concatenated_copy.geoh5",
            version=2.0,
            ga_version="4.2",
        ) as new_space:
            dh_group_copy = dh_group.copy(parent=new_space)

            compare_entities(
                dh_group_copy,
                dh_group,
                ignore=[
                    "_metadata",
                    "_parent",
                    "_data",
                    "_index",
                    "_drillholes_tables",
                ],
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


def test_add_data_raises_error_bad_key(tmp_path):
    workspace = Workspace(tmp_path / "test.geoh5")
    dh_group = DrillholeGroup.create(workspace)
    dh = Drillhole.create(workspace, name="dh1", parent=dh_group)
    msg = "Valid depth keys are 'depth' and 'from-to'"
    with pytest.raises(AttributeError, match=msg):
        dh.add_data(
            {"my data": {"depths": np.arange(0, 10.0), "values": np.random.randn(10)}}
        )


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


def test_locations(tmp_path):
    ws = Workspace(tmp_path / "test.geoh5")
    dh_group = DrillholeGroup.create(ws)
    dh = Drillhole.create(ws, name="dh", parent=dh_group)
    dh.add_data(
        {
            "my data": {
                "depth": np.arange(0, 10.0),
                "values": np.random.randn(10),
            },
        },
        property_group="my property group",
    )

    property_group = dh.fetch_property_group(name="my property group")
    assert np.allclose(property_group.locations, np.arange(0, 10.0))

    dh.add_data(
        {
            "my other data": {
                "from-to": np.c_[np.arange(0, 10.0), np.arange(1, 11.0)],
                "values": np.random.randn(10),
            }
        },
        property_group="my other property group",
    )
    property_group = dh.fetch_property_group(name="my other property group")
    assert np.allclose(
        property_group.locations, np.c_[np.arange(0, 10.0), np.arange(1, 11.0)]
    )


def test_is_collocated(tmp_path):
    ws = Workspace(tmp_path / "test.geoh5")
    dh_group = DrillholeGroup.create(ws)
    dh = Drillhole.create(ws, name="dh", parent=dh_group)
    property_group = dh.fetch_property_group(
        name="some uninitialized group", association="DEPTH"
    )
    assert not property_group.is_collocated(np.arange(0, 10.0), 0.01)
    dh.add_data(
        {
            "my data": {
                "depth": np.arange(0, 10.0),
                "values": np.random.randn(10),
            },
        },
        property_group="my property group",
    )
    property_group = dh.fetch_property_group(name="my property group")
    assert property_group.is_collocated(np.arange(0, 10.0), 0.01)
    assert property_group.is_collocated(np.arange(0.001, 10), 0.01)
    assert not property_group.is_collocated(np.arange(1, 11.0), 0.01)
    assert not property_group.is_collocated(np.arange(0, 9.0), 0.01)
    assert not property_group.is_collocated(
        np.c_[np.arange(0, 10.0), np.arange(1, 11.0)], 0.01
    )

    dh2 = Drillhole.create(ws, name="dh2", parent=dh_group)
    dh2.add_data(
        {
            "my other data": {
                "depth": np.arange(1, 11.0),
                "values": np.random.randn(10),
            },
        },
        property_group="my property group",
    )

    property_group = dh2.fetch_property_group(name="my property group")
    assert property_group.is_collocated(np.arange(1, 11.0), 0.01)

    dh.add_data(
        {
            "my other data": {
                "from-to": np.c_[np.arange(0, 10.0), np.arange(1, 11.0)],
                "values": np.random.randn(10),
            }
        },
        property_group="my other property group",
    )
    property_group = dh.fetch_property_group(name="my other property group")
    assert property_group.is_collocated(
        np.c_[np.arange(0, 10.0), np.arange(1, 11.0)], 0.01
    )


def compare_structured_arrays(
    array1: np.ndarray, array2: np.ndarray, tolerance: float = 1e-5
) -> bool:
    """
    Compare two NumPy record arrays.

    :param array1: The first structured array to compare.
    :param array2: The second array to be compared.
    :param tolerance: The tolerance level for numerical comparison.
    :param ignore: The names of the columns to ignore in the comparison.

    :return: True if arrays are equivalent within the given tolerance, False otherwise.
    """
    if array1.dtype.names != array2.dtype.names:
        return False  # Different structure

    for column in array1.dtype.names:
        data1, data2 = array1[column], array2[column]

        # Decode byte strings to regular strings if necessary
        if data1.dtype.kind == "S":
            data1 = np.array([d.decode() for d in data1])
            data2 = np.array([d.decode() for d in data2])

        # Check if the data type of the column is numerical
        if np.issubdtype(data1.dtype, np.number):
            if not np.all(np.isclose(data1, data2, atol=tolerance, equal_nan=True)):
                return False
        else:
            if not np.array_equal(data1, data2):
                return False

    return True


def test_export_table(tmp_path):
    h5file_path = tmp_path / r"test_drillholeGroup.geoh5"
    _, workspace = create_drillholes(h5file_path, version=2.0, ga_version="4.2")

    with workspace.open():
        drillhole_group = workspace.get_entity("DH_group")[0]
        n_ndv = 25
        values = [
            np.array(
                [["{%s}" % child.uid] * 25 for child in drillhole_group.children[::2]]
            )
            .flatten()
            .astype("S"),
            drillhole_group.data["FROM"],
            drillhole_group.data["TO"],
            np.array(
                drillhole_group.data["interval_values_a"].tolist() + [np.nan] * n_ndv
            ),
            np.array(
                [np.nan] * n_ndv + drillhole_group.data["interval_values_b"].tolist()
            ),
            np.array(drillhole_group.data["text Data"].tolist() + [""] * n_ndv),
        ]

        dtypes = [
            ("Drillhole", "O"),
            ("FROM", np.float64),
            ("TO", np.float64),
            ("interval_values_a", np.float64),
            ("interval_values_b", np.float64),
            ("text Data", "O"),
        ]

        verification = np.core.records.fromarrays(values, dtype=dtypes)

        assert compare_structured_arrays(
            drillhole_group.drillholes_tables["property_group"].depth_table,
            verification,
            tolerance=1e-5,
        )

        values = [
            np.array(
                [["{%s}" % child.uid] * 50 for child in drillhole_group.children[:-2]]
            )
            .flatten()
            .astype("S"),
            drillhole_group.data["DEPTH"],
            drillhole_group.data["my_log_values/"],
        ]

        dtypes = [
            ("Drillhole", "O"),
            ("DEPTH", np.float64),
            ("my_log_values/", np.float64),
        ]

        verification = np.core.records.fromarrays(values, dtype=dtypes)

        # drillholes cannot be compared directly as the order is based on depth
        assert compare_structured_arrays(
            drillhole_group.drillholes_tables["depth_0"].depth_table_by_name(
                "my_log_values/", spatial_index=True
            )[["DEPTH", "my_log_values/"]],
            verification[["DEPTH", "my_log_values/"]],
            tolerance=1e-5,
        )

        assert compare_structured_arrays(
            drillhole_group.drillholes_tables["depth_0"].depth_table_by_name(
                "my_log_values/", spatial_index=False
            ),
            verification[["my_log_values/"]],
            tolerance=1e-5,
        )


def test_add_data_to_property(tmp_path):
    h5file_path = tmp_path / r"test_drillholeGroup.geoh5"
    drillhole_group, workspace = create_drillholes(
        h5file_path, version=2.0, ga_version="4.2"
    )

    with workspace.open():
        drillholes_table = drillhole_group.drillholes_tables["property_group"]
        verification = drillholes_table.depth_table_by_name(
            "interval_values_a", spatial_index=True
        )

        verification_map_value = np.random.randint(
            0, 100, verification["interval_values_a"].shape[0], dtype=np.int32
        )
        value_map = {idx: f"abc{idx}" for idx in np.unique(verification_map_value)}
        value_map[0] = "Unknown"

        drillholes_table.add_values_to_property_group(
            "new value int", verification_map_value
        )

        drillholes_table.add_values_to_property_group(
            "new value",
            verification_map_value,
            data_type=data_type.ReferencedValueMapType(
                workspace,
                primitive_type="REFERENCED",
                name="new_value",
                value_map=value_map,
            ),
        )

        verificationb = drillholes_table.depth_table_by_name("new value")
        verificatione = drillholes_table.depth_table_by_name("new value int")
        verificatione.dtype.names = verificationb.dtype.names

        assert compare_structured_arrays(
            verificationb,
            verificatione,
            tolerance=1e-5,
        )

        # reopen
        drillhole_group = workspace.get_entity("DH_group")[0]

        verification = drillhole_group.drillholes_tables[
            "property_group"
        ].depth_table_by_name("interval_values_a")

        verificationd = drillhole_group.drillholes_tables[
            "property_group"
        ].depth_table_by_name("new value")

        assert compare_structured_arrays(
            verificationd,
            verificationb,
            tolerance=1e-5,
        )

        # change the name for the verification
        verificationd.dtype.names = verification.dtype.names
        verificationc = verification.copy()
        verificationc["interval_values_a"] = verification_map_value

        assert compare_structured_arrays(
            verificationc.astype([("new_value", np.float32)]),
            verificationd.astype([("new_value", np.float32)]),
            tolerance=1e-5,
        )

        test_drillhole_table = drillhole_group.drillholes_table_from_data_name[
            "interval_values_a"
        ]

        assert compare_structured_arrays(
            test_drillhole_table.depth_table_by_name("interval_values_a"),
            verification,
            tolerance=1e-5,
        )

        verificationf = drillhole_group.drillholes_tables[
            "property_group"
        ].depth_table_by_name("new value", mapped=True)

        assert verificationf[0][0] == value_map[verification_map_value[0]]


def test_tables_errors(tmp_path):
    h5file_path = tmp_path / r"test_drillholeGroup.geoh5"

    drillhole_group, workspace = create_drillholes(
        h5file_path, version=2.0, ga_version="4.2"
    )

    with workspace.open():
        with pytest.raises(TypeError, match="The parent must be a Concatenator"):
            DrillholesGroupTable._get_property_groups("bidon", "bidon")

        with pytest.raises(ValueError, match="No property group with name"):
            DrillholesGroupTable._get_property_groups(drillhole_group, "bidon")

        with pytest.raises(KeyError, match="The name must"):
            drillhole_group.drillholes_tables[
                "property_group"
            ].add_values_to_property_group(name=123, values=np.random.randn(50))

        with pytest.raises(ValueError, match="The length of the values"):
            drillhole_group.drillholes_tables[
                "property_group"
            ].add_values_to_property_group(name="new value", values=np.random.randn(49))

        with pytest.raises(KeyError, match="The names are not in the list"):
            drillhole_group.drillholes_tables["property_group"].depth_table_by_name(
                ("bidon", "bidon")
            )

        with pytest.raises(KeyError, match="The name 'bidon' is not in"):
            drillhole_group.drillholes_tables["property_group"].nan_value_from_name(
                "bidon"
            )


def test_surveys_info(tmp_path):
    h5file_path = tmp_path / r"test_info.geoh5"

    _, workspace = create_drillholes(h5file_path, version=2.0, ga_version="1.0")

    # Manually add an info column to the surveys
    dh_group = workspace.get_entity("DH_group")[0]

    surveys = dh_group.data["Surveys"].view("<f4").reshape((-1, 3))

    new_dtype = [
        ("Depth", "<f4"),
        ("Azimuth", "<f4"),
        ("Dip", "<f4"),
        ("Info", special_dtype(vlen=str)),
    ]
    values = []

    for val in surveys:
        values.append((*val, b""))

    new_surveys = np.array(values, dtype=new_dtype)

    dh_group.data["Surveys"] = new_surveys

    with workspace.open(mode="r+"):
        dh_group.save_attribute("surveys")

    with workspace.open(mode="r+") as workspace:
        dh_group = workspace.get_entity("DH_group")[0]
        dh = Drillhole.create(
            workspace,
            collar=np.r_[0.0, 10.0, 10],
            surveys=np.c_[
                np.linspace(0, 100, 5),
                np.ones(5) * 45.0,
                np.ones(5) * -89.0,
            ],
            parent=dh_group,
            name="Info Drillhole",
        )

    assert len(dh.parent.data["Surveys"]) == 35
    assert "Info" in dh.parent.data["Surveys"].dtype.names

    with workspace.open():
        dh = workspace.get_entity("Info Drillhole")[0]
        assert len(dh.surveys) == 5
