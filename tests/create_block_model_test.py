import os

import numpy as np

from geoh5io.objects import BlockModel
from geoh5io.workspace import Workspace


def test_create_block_model_data():

    h5file = r"temp\block_model.geoh5"
    name = "MyTestBlockModel"

    # Generate a 3D array
    n_x, n_y, n_z = 8, 9, 10

    nodal_x = np.r_[
        0,
        np.cumsum(
            np.r_[
                np.pi / n_x * 1.5 ** np.arange(3)[::-1],
                np.ones(n_x) * np.pi / n_x,
                np.pi / n_x * 1.5 ** np.arange(4),
            ]
        ),
    ]
    nodal_y = np.r_[
        0,
        np.cumsum(
            np.r_[
                np.pi / n_y * 1.5 ** np.arange(5)[::-1],
                np.ones(n_y) * np.pi / n_y,
                np.pi / n_y * 1.5 ** np.arange(6),
            ]
        ),
    ]
    nodal_z = -np.r_[
        0,
        np.cumsum(
            np.r_[
                np.pi / n_z * 1.5 ** np.arange(7)[::-1],
                np.ones(n_z) * np.pi / n_z,
                np.pi / n_z * 1.5 ** np.arange(8),
            ]
        ),
    ]

    origin = [0, 0, 0]

    # # Create a workspace
    workspace = Workspace(os.getcwd() + os.sep + "assets" + os.sep + h5file)

    grid = BlockModel.create(
        workspace,
        origin=origin,
        u_cell_delimiters=nodal_x,
        v_cell_delimiters=nodal_y,
        z_cell_delimiters=nodal_z,
        name=name,
        rotation=30,
        allow_move=False,
    )

    values = (
        np.cos(grid.centroids[:, 0])
        * np.cos(grid.centroids[:, 1])
        * np.cos(grid.centroids[:, 2])
    )
    grid.add_data({"DataValues": {"association": "CELL", "values": values}})

    # Read the data back in from a fresh workspace
    workspace = Workspace(os.getcwd() + os.sep + "assets" + os.sep + h5file)

    obj = workspace.get_entity(name)[0]

    data = workspace.get_entity("DataValues")[0]

    assert all(data.values == values), "Data values differ from input"
    assert all(
        (obj.centroids == grid.centroids).flatten()
    ), "Data locations differ from input"

    os.remove(os.getcwd() + os.sep + "assets" + os.sep + h5file)
