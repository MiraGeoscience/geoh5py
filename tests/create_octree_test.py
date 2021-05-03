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

import numpy as np

from geoh5py.objects import Octree
from geoh5py.workspace import Workspace


def test_octree():

    name = "MyTestOctree"

    with tempfile.TemporaryDirectory() as tempdir:
        h5file_path = Path(tempdir) / r"octree.geoh5"

        # Create a workspace
        workspace = Workspace(h5file_path)

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

        assert mesh.n_cells == 8, "Number of octree cells after base_refine is wrong"

        # Refine
        workspace.save_entity(mesh)
        workspace.finalize()

        # Read the mesh back in
        workspace = Workspace(h5file_path)
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
