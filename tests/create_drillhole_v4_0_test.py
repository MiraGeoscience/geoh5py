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


import numpy as np
import pytest

from geoh5py.groups import DrillholeGroup
from geoh5py.objects import Drillhole
from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace


def test_create_drillhole_data(tmp_path):
    h5file_path = tmp_path / r"testCurve.geoh5"
    well_name = "bullseye"
    n_data = 10

    with Workspace(h5file_path, version=2.0) as workspace:
        # Create a workspace
        dh_group = DrillholeGroup.create(workspace)

        assert (
            dh_group.data == {}
        ), "DrillholeGroup should have an empty dictionary of data."
        with pytest.raises(AttributeError) as excinfo:
            dh_group.concatenated_attributes = []

        assert "Input value for 'concatenated_attributes' must be of type dict." in str(
            excinfo
        )

        with pytest.raises(AttributeError) as excinfo:
            dh_group.concatenated_attributes = {}

        assert (
            "The first key of 'concatenated_attributes' must be 'Attributes'."
            in str(excinfo)
        )

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
                    "from-to": from_to_a,
                },
                "int_interval_list": {
                    "values": np.asarray([1, 2, 3]),
                    "from-to": from_to_b,
                    "value_map": {1: "Unit_A", 2: "Unit_B", 3: "Unit_C"},
                    "type": "referenced",
                },
                "interval_values_b": {
                    "values": np.random.randn(from_to_b.shape[0]),
                    "from-to": from_to_b,
                },
            }
        )

        well_b_data = well_b.add_data(
            {
                "interval_values": {
                    "values": np.random.randn(from_to_b.shape[0]),
                    "from-to": from_to_b,
                },
            }
        )

        assert dh_group.fetch_index(well_b_data, well_b_data.name) == 1, (
            "'interval_values' on well_b should be the second entry.",
        )

        well_b_data.values = np.random.randn(from_to_b.shape[0])

    new_workspace = Workspace(h5file_path)
    # Check entities
    rec_well = new_workspace.get_entity(well_name)[0]
    compare_entities(
        well,
        rec_well,
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

    compare_entities(
        well_b,
        new_workspace.get_entity("Number 2")[0],
        ignore=[
            "_parent",
            "_metadata",
            "_default_collocation_distance",
            "_property_groups",
        ],
    )
    rec_well = new_workspace.get_entity("Number 2")[0]
    rec_well.get_data_list()
    compare_entities(
        well_b_data,
        rec_well.get_data("interval_values")[0],
        ignore=["_metadata", "_parent"],
    )

    assert rec_well.get_data_list() == well_b.get_data_list()
