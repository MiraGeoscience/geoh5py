import os

import numpy.random as random

from geoh5io.objects import Curve
from geoh5io.workspace import Workspace


def test_create_point_data():

    h5file = "testCurve.geoh5"

    # Generate a random cloud of points
    n_data = 12
    xyz = random.randn(n_data, 3)
    values = random.randn(n_data)
    cell_values = random.randn(n_data - 1)

    # Create a workspace
    workspace = Workspace(r".\assets" + os.sep + h5file)

    curve, _, _ = Curve.create(
        workspace,
        xyz,
        data={"vertexValues": ["VERTEX", values], "cellValues": ["CELL", cell_values]},
    )
    curve.save_to_h5()
    workspace.finalize()

    # Read the data back in
    obj_list = workspace.list_objects

    obj = workspace.get_entity(obj_list[0])[0]
    data_vertex = workspace.get_entity(obj.get_data_list[0])[0]
    data_cell = workspace.get_entity(obj.get_data_list[1])[0]

    assert all(data_vertex.values == values), "VERTEX data values differ from input"
    assert all(data_cell.values == cell_values), "VERTEX data values differ from input"
    assert all((obj.vertices() == xyz).flatten()), "Data locations differ from input"

    os.remove(r".\assets" + os.sep + h5file)
