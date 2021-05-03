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

import numpy.random as random

from geoh5py.objects import Curve
from geoh5py.workspace import Workspace


def test_create_curve_data():

    curve_name = "TestCurve"

    # Generate a random cloud of points
    n_data = 12
    xyz = random.randn(n_data, 3)
    values = random.randn(n_data)
    cell_values = random.randn(n_data - 1)

    with tempfile.TemporaryDirectory() as tempdir:
        h5file_path = Path(tempdir) / r"testCurve.geoh5"

        # Create a workspace
        workspace = Workspace(h5file_path)

        curve = Curve.create(workspace, vertices=xyz, name=curve_name)

        data_objects = curve.add_data(
            {"vertexValues": {"values": values}, "cellValues": {"values": cell_values}}
        )
        workspace.finalize()

        assert all(
            data_objects[0].values == values
        ), "Created VERTEX data values differ from input"
        assert all(
            data_objects[1].values == cell_values
        ), "Created CELL data values differ from input"

        #################### Modify the vertices and data #########################
        # Re-open the workspace and read data back in
        workspace = Workspace(h5file_path)

        obj_rec = workspace.get_entity(curve_name)[0]
        assert all(
            (obj_rec.vertices == xyz).flatten()
        ), "Data locations differ from input"

        for attr in obj_rec.attribute_map.values():
            assert getattr(obj_rec, attr) == getattr(
                curve, attr
            ), f"Attribute {attr} do not match output"

        data_vertex = workspace.get_entity("vertexValues")[0]
        data_cell = workspace.get_entity("cellValues")[0]

        assert all(
            data_vertex.values == values
        ), "Loaded VERTEX data values differ from input"
        assert all(
            data_cell.values == cell_values
        ), "Loaded CELL data values differ from input"

        # Change the vertices of the curve
        xyz = random.randn(n_data, 3)
        obj_rec.vertices = xyz

        # Change the vertex values
        values = random.randn(n_data)
        data_vertex.values = values

        workspace.finalize()

        ##################### READ BACK AND COMPARE ############################
        workspace = Workspace(h5file_path)

        # Read the data back in again
        obj = workspace.get_entity(curve_name)[0]
        assert all(
            (obj.vertices == xyz).flatten()
        ), "Modified data locations differ from input"

        data_vertex = workspace.get_entity("vertexValues")[0]

        assert all(
            data_vertex.values == values
        ), "Modified VERTEX data values differ from input"
