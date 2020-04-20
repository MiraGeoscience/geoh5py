import os

import numpy as np

from geoh5py.objects import Octree
from geoh5py.workspace import Workspace


def test_octree():

    h5file = r"octree.geoh5"
    name = "MyTestOctree"

    # Create a workspace
    workspace = Workspace(os.getcwd() + os.sep + "assets" + os.sep + h5file)

    # Create an octree mesh with variable dimensions
    mesh = Octree.create(
        workspace,
        name=name,
        origin=[0, 0, 0],
        u_count=32,
        v_count=16,
        w_count=8,
        u_cell_size=1.0,
        v_cell_size=1.0,
        w_cell_size=2.0,
        rotation=45,
    )

    # Refine globally to first octree
    # mesh.refine(1)

    assert mesh.n_cells == 8, "Number of octree cells after refine is wrong"

    # Refine
    workspace.save_entity(mesh)
    workspace.finalize()

    # Read the mesh back in
    workspace = Workspace(os.getcwd() + os.sep + "assets" + os.sep + h5file)

    mesh2 = workspace.get_entity(name)[0]

    assert all(
        mesh2.octree_cells == mesh.octree_cells
    ), "Mesh output differs from mesh input"
    assert all(
        np.r_[mesh2.origin] == np.r_[mesh.origin]
    ), "Mesh output differs from mesh input"

    assert all(
        np.r_[mesh2.rotation] == np.r_[mesh.rotation]
    ), "Mesh output differs from mesh input"

    os.remove(os.getcwd() + os.sep + "assets" + os.sep + h5file)
