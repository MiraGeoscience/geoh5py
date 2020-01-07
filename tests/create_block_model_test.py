import os

import numpy as np

from geoh5io.objects import BlockModel
from geoh5io.workspace import Workspace


def test_create_block_model_data():

    h5file = r"temp\block_model.geoh5"
    name = "MyTestBlockModel"

    # Generate a 2D array
    n_x, n_y, n_z = 8, 9, 10
    pad_x, pad_y, pad_z = [3, 4], [3, 5], [0, 6]
    d_x, d_y, d_z = np.pi / n_x, np.pi / n_y, np.pi / n_z

    nodal_x = np.r_[
        0,
        np.cumsum(
            np.r_[
                d_x * 1.5 ** np.arange(pad_x[0])[::-1],
                np.ones(n_x) * d_x,
                d_x * 1.5 ** np.arange(pad_x[1]),
            ]
        ),
    ]
    nodal_y = np.r_[
        0,
        np.cumsum(
            np.r_[
                d_y * 1.5 ** np.arange(pad_y[0])[::-1],
                np.ones(n_y) * d_y,
                d_y * 1.5 ** np.arange(pad_y[1]),
            ]
        ),
    ]
    nodal_z = -np.r_[
        0,
        np.cumsum(
            np.r_[
                d_z * 1.5 ** np.arange(pad_z[0])[::-1],
                np.ones(n_z) * d_z,
                d_z * 1.5 ** np.arange(pad_z[1]),
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
    data = grid.add_data({"DataValues": ["CELL", values]})
    workspace.save_entity(grid)
    workspace.finalize()

    # Read the data back in from a fresh workspace
    workspace = Workspace(os.getcwd() + os.sep + "assets" + os.sep + h5file)

    obj = workspace.get_entity(name)[0]

    data = workspace.get_entity("DataValues")[0]

    assert all(data.values == values), "Data values differ from input"
    assert all(
        (obj.centroids == grid.centroids).flatten()
    ), "Data locations differ from input"

    os.remove(os.getcwd() + os.sep + "assets" + os.sep + h5file)
