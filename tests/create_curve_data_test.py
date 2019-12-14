import os

import numpy.random as random

from geoh5io.objects import Curve
from geoh5io.workspace import Workspace


def test_create_curve_data():

    h5file = r"temp\testCurve.geoh5"
    curve_name = "TestCurve"

    # Generate a random cloud of points
    n_data = 12
    xyz = random.randn(n_data, 3)
    values = random.randn(n_data)
    cell_values = random.randn(n_data - 1)

    # Create a workspace
    workspace = Workspace(os.getcwd() + os.sep + "assets" + os.sep + h5file)

    curve, vertex_data, cell_data = Curve.create(
        workspace,
        vertices=xyz,
        name=curve_name,
        data={"vertexValues": ["VERTEX", values], "cellValues": ["CELL", cell_values]},
    )

    assert all(
        vertex_data.values == values
    ), "Created VERTEX data values differ from input"
    assert all(
        cell_data.values == cell_values
    ), "Created CELL data values differ from input"

    workspace.save_entity(curve)
    workspace.finalize()

    # Read the data back in
    obj_list = workspace.list_objects

    obj = workspace.get_entity(obj_list[0])[0]
    assert all((obj.vertices() == xyz).flatten()), "Data locations differ from input"

    data_vertex = workspace.get_entity("vertexValues")[0]
    data_cell = workspace.get_entity("cellValues")[0]

    assert all(
        data_vertex.values == values
    ), "Loaded VERTEX data values differ from input"
    assert all(
        data_cell.values == cell_values
    ), "Loaded CELL data values differ from input"

    #################### Modify the vertices and data #########################
    # Change the vertices of the curve
    new_locs = random.randn(n_data, 3)
    curve.vertices = new_locs

    # Change the vertex values
    new_vals = random.randn(n_data)
    vertex_data.values = new_vals
    workspace.save_entity(curve)

    # Read the data back in
    obj = workspace.get_entity(curve_name)[0]
    assert all(
        (obj.vertices() == new_locs).flatten()
    ), "Modified data locations differ from input"

    data_vertex = workspace.get_entity("vertexValues")[0]
    data_cell = workspace.get_entity("cellValues")[0]

    assert all(
        data_vertex.values == new_vals
    ), "Modified VERTEX data values differ from input"

    os.remove(os.getcwd() + os.sep + "assets" + os.sep + h5file)
