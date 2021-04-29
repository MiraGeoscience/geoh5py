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

import random
import string
import tempfile
from pathlib import Path

import numpy as np

from geoh5py.objects import Drillhole
from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace


def test_create_drillhole_data():

    well_name = "bullseye"
    n_data = 10

    with tempfile.TemporaryDirectory() as tempdir:
        h5file_path = Path(tempdir) / r"testCurve.geoh5"
        # Create a workspace
        workspace = Workspace(h5file_path)
        max_depth = 100
        well = Drillhole.create(
            workspace,
            collar=np.r_[0.0, 10.0, 10],
            surveys=np.c_[
                np.linspace(0, max_depth, n_data),
                np.linspace(-89, -75, n_data),
                np.ones(n_data) * 45.0,
            ],
            name=well_name,
        )
        value_map = {}
        for ref in range(8):
            value_map[ref] = "".join(
                random.choice(string.ascii_lowercase) for i in range(8)
            )

        # Create random from-to
        from_to_a = np.sort(
            np.random.uniform(low=0.0, high=max_depth, size=(50,))
        ).reshape((-1, 2))
        from_to_b = np.vstack([from_to_a[0, :], [30.1, 55.5], [56.5, 80.2]])

        # Add from-to data
        data_objects = well.add_data(
            {
                "interval_values": {
                    "values": np.random.randn(from_to_a.shape[0]),
                    "from-to": from_to_a,
                },
                "int_interval_list": {
                    "values": [1, 2, 3],
                    "from-to": from_to_b,
                    "value_map": {1: "Unit_A", 2: "Unit_B", 3: "Unit_C"},
                    "type": "referenced",
                },
            }
        )

        assert well.n_cells == (
            from_to_a.shape[0] + 2
        ), "Error with number of cells on interval data creation."
        assert well.n_vertices == (
            from_to_a.size + 4
        ), "Error with number of vertices on interval data creation."
        assert not np.any(
            np.isnan(well.get_data("FROM")[0].values)
        ), "FROM values not fully set."
        assert not np.any(
            np.isnan(well.get_data("TO")[0].values)
        ), "TO values not fully set."
        assert (
            well.get_data("TO")[0].values.shape[0]
            == well.get_data("FROM")[0].values.shape[0]
            == well.n_cells
        ), "Shape or FROM to n_cells differ."

        # Add log-data
        data_objects += [
            well.add_data(
                {
                    "log_values": {
                        "depth": np.sort(np.random.rand(n_data) * max_depth),
                        "type": "referenced",
                        "values": np.random.randint(1, high=8, size=n_data),
                        "value_map": value_map,
                    }
                }
            )
        ]
        workspace.finalize()

        assert well.n_vertices == (
            from_to_a.size + 4 + n_data
        ), "Error with new number of vertices on log data creation."
        # Re-open the workspace and read data back in
        new_workspace = Workspace(h5file_path)
        # Check entities
        compare_entities(well, new_workspace.get_entity(well_name)[0])
        compare_entities(
            data_objects[0], new_workspace.get_entity("interval_values")[0]
        )
        compare_entities(
            data_objects[1], new_workspace.get_entity("int_interval_list")[0]
        )
        compare_entities(data_objects[2], new_workspace.get_entity("log_values")[0])
