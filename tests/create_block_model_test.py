import tempfile
from pathlib import Path

import numpy as np

from geoh5py.objects import BlockModel
from geoh5py.shared import Entity, EntityType
from geoh5py.workspace import Workspace


def compare_objects(object_a, object_b):
    for attr in object_a.__dict__.keys():
        if attr in ["_workspace", "_children"]:
            continue
        if isinstance(getattr(object_a, attr[1:]), (Entity, EntityType)):
            compare_objects(getattr(object_a, attr[1:]), getattr(object_b, attr[1:]))
        else:
            assert np.all(
                getattr(object_a, attr[1:]) == getattr(object_b, attr[1:])
            ), f"Output attribute {attr[1:]} for {object_a} do not match input {object_b}"


def test_create_block_model_data():

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

    with tempfile.TemporaryDirectory() as tempdir:
        h5file_path = Path(tempdir) / r"block_model.geoh5"

        # Create a workspace
        workspace = Workspace(h5file_path)

        grid = BlockModel.create(
            workspace,
            origin=[0, 0, 0],
            u_cell_delimiters=nodal_x,
            v_cell_delimiters=nodal_y,
            z_cell_delimiters=nodal_z,
            name=name,
            rotation=30,
            allow_move=False,
        )

        data = grid.add_data(
            {
                "DataValues": {
                    "association": "CELL",
                    "values": (
                        np.cos(grid.centroids[:, 0])
                        * np.cos(grid.centroids[:, 1])
                        * np.cos(grid.centroids[:, 2])
                    ),
                }
            }
        )

        # Read the data back in from a fresh workspace
        workspace = Workspace(h5file_path)

        rec_obj = workspace.get_entity(name)[0]
        rec_data = workspace.get_entity("DataValues")[0]

        compare_objects(grid, rec_obj)
        compare_objects(data, rec_data)
