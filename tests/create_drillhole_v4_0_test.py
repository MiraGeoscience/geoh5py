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

# pylint: disable=R0914

import numpy as np
import pytest

from geoh5py.data import FloatData, data_type
from geoh5py.groups import ContainerGroup, DrillholeGroup
from geoh5py.objects import Drillhole
from geoh5py.shared.concatenation import (
    ConcatenatedData,
    ConcatenatedObject,
    ConcatenatedPropertyGroup,
    Concatenator,
)
from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace


def test_concatenator(tmp_path):

    h5file_path = tmp_path / r"test_Concatenator.geoh5"

    with Workspace(h5file_path, version=2.0) as workspace:
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
    with Workspace(h5file_path, version=2.0) as workspace:

        class_type = type("TestGroup", (Concatenator, ContainerGroup), {})
        entity_type = class_type.find_or_create_type(workspace)
        concat = class_type(entity_type)

        class_type = type("TestObject", (ConcatenatedObject, Drillhole), {})
        entity_type = class_type.find_or_create_type(workspace)

        with pytest.raises(UserWarning) as error:
            concat_object = class_type(entity_type)

        assert (
            "Creating a concatenated object must have a parent of type Concatenator."
            in str(error)
        )

        concat_object = class_type(entity_type, parent=concat)

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
            prop_group = ConcatenatedPropertyGroup()

        assert "Creating a concatenated data must have a parent" in str(error)

        prop_group = ConcatenatedPropertyGroup(parent=concat_object)

        with pytest.raises(AttributeError) as error:
            prop_group.parent = Drillhole

        assert (
            "The 'parent' of a concatenated Data must be of type 'Concatenated'"
            in str(error)
        )

        assert prop_group.to_ is None
        assert prop_group.from_ is None


def test_create_drillhole_data(tmp_path):
    h5file_path = tmp_path / r"test_drillholeGroup.geoh5"
    new_path = tmp_path / r"test_drillholeGroup2.geoh5"
    well_name = "bullseye/"
    n_data = 10

    with Workspace(h5file_path, version=2.0) as workspace:
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

        with pytest.raises(UserWarning) as error:
            dh_group.update_array_attribute(well, "abc")

        assert f"Input entity {well} does not have a property or values" in str(error)

        # Add both set of log data with 0.5 m tolerance
        values = np.random.randn(50)
        with pytest.raises(UserWarning) as error:
            well.add_data(
                {
                    "my_log_values/": {
                        "depth": np.arange(0, 50.0),
                        "values": values,
                    }
                },
                collocation_distance=-1.0,
            )

        assert "Input depth 'collocation_distance' must be >0." in str(error)

        # Add both set of log data with 0.5 m tolerance
        with pytest.raises(AttributeError) as error:
            well.add_data(
                {
                    "my_log_values/": {
                        "values": np.random.randn(50),
                    }
                },
            )

        assert "Input data dictionary must contain" in str(error)

        well.add_data(
            {
                "my_log_values/": {
                    "depth": np.arange(0, 50.0),
                    "values": np.random.randn(50),
                },
                "log_wt_tolerance": {
                    "depth": np.arange(0.01, 50.01),
                    "values": np.random.randn(50),
                },
            }
        )

        assert len(well.get_data("my_log_values/")) == 1

        with pytest.raises(UserWarning) as error:
            well.add_data(
                {
                    "my_log_values/": {
                        "depth": np.arange(0, 50.0),
                        "values": np.random.randn(50),
                    },
                }
            )

        assert "already present on the drillhole" in str(error)

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
                    "values": np.random.randn(10),
                    "depth": np.sort(np.random.uniform(low=0.05, high=100, size=(10,))),
                },
            }
        )

        assert dh_group.fetch_index(well_b_data, well_b_data.name) == 1, (
            "'interval_values' on well_b should be the second entry.",
        )

        assert len(well.to_) == len(well.from_) == 3, "Should have only 3 from-to data."

        with pytest.raises(UserWarning) as error:
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

        assert "Data with name 'Depth Data' already present" in str(error)

        well_b_data.values = np.random.randn(from_to_b.shape[0])

    with well.workspace.open() as workspace:
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

        with Workspace(new_path, version=2.0) as new_workspace:
            new_group = dh_group.copy(parent=new_workspace)
            well = new_group.children[0]

            with pytest.raises(ValueError) as error:
                well.add_data(
                    {
                        "new_data": {"values": np.random.randn(49).astype(np.float32)},
                    },
                    property_group=well.property_groups[0].name,
                )

            assert "Input values for 'new_data' with shape(49)" in str(error)

            well.add_data(
                {
                    "new_data": {"values": np.random.randn(50).astype(np.float32)},
                },
                property_group=well.property_groups[0].name,
            )

        assert (
            len(well.property_groups[0].properties) == 5
        ), "Issue adding data to interval."
