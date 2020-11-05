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
from abc import ABC
from pathlib import Path

import numpy as np
from geoh5py.objects import Curve
from geoh5py.workspace import Workspace


def test_create_curve_data():

    curve_name = "TestCurve"

    # Generate a random cloud of points
    n_data = 12

    def compare_objects(object_a, object_b):
        for attr in object_a.__dict__.keys():
            if attr in ["_workspace", "_children"]:
                continue
            if isinstance(getattr(object_a, attr[1:]), ABC):
                compare_objects(
                    getattr(object_a, attr[1:]), getattr(object_b, attr[1:])
                )
            else:
                assert np.all(
                    getattr(object_a, attr[1:]) == getattr(object_b, attr[1:])
                ), f"Output attribute {attr[1:]} for {object_a} do not match input {object_b}"

    with tempfile.TemporaryDirectory() as tempdir:

        h5file_path = Path(tempdir) / r"testCurve.geoh5"

        # Create a workspace
        workspace = Workspace(h5file_path)

        curve = Curve.create(
            workspace, vertices=np.random.randn(n_data, 3), name=curve_name
        )

        # Get and change the parts
        parts = curve.parts
        parts[-3:] = 1
        curve.parts = parts

        data_objects = curve.add_data(
            {
                "vertexValues": {"values": np.random.randn(curve.n_vertices)},
                "cellValues": {"values": np.random.randn(curve.n_cells)},
            }
        )

        workspace.finalize()
        # Re-open the workspace and read data back in
        workspace = Workspace(h5file_path)

        obj_rec = workspace.get_entity(curve_name)[0]
        data_vert_rec = workspace.get_entity("vertexValues")[0]
        data_cell_rec = workspace.get_entity("cellValues")[0]

        # Check entities
        compare_objects(curve, obj_rec)
        compare_objects(data_objects[0], data_vert_rec)
        compare_objects(data_objects[1], data_cell_rec)

        # Modify and write
        obj_rec.vertices = np.random.randn(n_data, 3)
        data_vert_rec.values = np.random.randn(n_data)
        workspace.finalize()

        # Read back and compare
        workspace = Workspace(h5file_path)
        obj = workspace.get_entity(curve_name)[0]
        data_vertex = workspace.get_entity("vertexValues")[0]

        compare_objects(obj_rec, obj)
        compare_objects(data_vert_rec, data_vertex)
