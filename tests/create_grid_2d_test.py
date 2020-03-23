import os

import numpy as np

from geoh5io.objects import Grid2D
from geoh5io.shared import Entity, EntityType
from geoh5io.workspace import Workspace


def test_create_grid_2d_data():

    h5file = r"test2Grid.geoh5"
    name = "MyTestGrid2D"

    # Generate a 2D array
    n_x, n_y = 10, 15
    d_x, d_y = 20.0, 30.0
    origin = [0, 0, 0]
    values, _ = np.meshgrid(np.linspace(0, np.pi, n_x), np.linspace(0, np.pi, n_y))

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

    data = grid.add_data({"DataValues": {"values": values}})
    grid.rotation = 45.0

    workspace.finalize()

    # Read the data back in from a fresh workspace
    workspace = Workspace(os.getcwd() + os.sep + "assets" + os.sep + h5file)

    rec_obj = workspace.get_entity(name)[0]

    rec_data = workspace.get_entity("DataValues")[0]

    def compare_objects(object_a, object_b):
        for attr in object_a.__dict__.keys():
            if attr in ["_workspace", "_children"]:
                continue
            if isinstance(getattr(object_a, attr[1:]), (Entity, EntityType)):
                compare_objects(
                    getattr(object_a, attr[1:]), getattr(object_b, attr[1:])
                )
            else:
                assert np.all(
                    getattr(object_a, attr[1:]) == getattr(object_b, attr[1:])
                ), f"Output attribute {attr[1:]} for {object_a} do not match input {object_b}"

    compare_objects(grid, rec_obj)
    compare_objects(data, rec_data)

    os.remove(os.getcwd() + os.sep + "assets" + os.sep + h5file)
