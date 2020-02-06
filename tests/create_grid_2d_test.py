import os

from numpy import linspace, meshgrid, pi, ravel

from geoh5io.objects import Grid2D
from geoh5io.workspace import Workspace


def test_create_grid_2d_data():

    h5file = r"temp\test2Grid.geoh5"
    name = "MyTestGrid2D"

    # Generate a 2D array
    n_x, n_y = 10, 15
    d_x, d_y = 20.0, 30.0
    origin = [0, 0, 0]
    values, _ = meshgrid(linspace(0, pi, n_x), linspace(0, pi, n_y))

    # # Create a workspace
    workspace = Workspace(os.getcwd() + os.sep + "assets" + os.sep + h5file)

    grid = Grid2D.create(
        workspace,
        origin=origin,
        u_cell_size=d_x,
        v_cell_size=d_y,
        u_count=n_x,
        v_count=n_y,
        name=name,
        allow_move=False,
    )

    grid.add_data({"DataValues": ["CELL", values]})
    grid.rotation = 45.0

    # Read the data back in from a fresh workspace
    workspace = Workspace(os.getcwd() + os.sep + "assets" + os.sep + h5file)

    # obj = workspace.get_entity(name)[0]

    data = workspace.get_entity("DataValues")[0]

    assert all(data.values == ravel(values)), "Data values differ from input"
    # assert all((obj.locations == xyz).flatten()), "Data locations differ from input"

    os.remove(os.getcwd() + os.sep + "assets" + os.sep + h5file)
