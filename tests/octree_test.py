# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2025-2026 Mira Geoscience Ltd.                                '
#                                                                              '
#  This file is part of geoh5py.                                               '
#                                                                              '
#  geoh5py is free software: you can redistribute it and/or modify             '
#  it under the terms of the GNU Lesser General Public License as published by '
#  the Free Software Foundation, either version 3 of the License, or           '
#  (at your option) any later version.                                         '
#                                                                              '
#  geoh5py is distributed in the hope that it will be useful,                  '
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              '
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               '
#  GNU Lesser General Public License for more details.                         '
#                                                                              '
#  You should have received a copy of the GNU Lesser General Public License    '
#  along with geoh5py.  If not, see <https://www.gnu.org/licenses/>.           '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from geoh5py.objects import Octree
from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace


def test_octree_cell_volumes(tmp_path: Path):
    with Workspace.create(tmp_path / "test.geoh5") as workspace:
        mesh = Octree.create(
            workspace,
            name="test",
            origin=[0, 0, 0],
            u_count=8,
            v_count=8,
            w_count=8,
            u_cell_size=10.0,
            v_cell_size=20.0,
            w_cell_size=40.0,
            octree_cells=np.array(
                [
                    [0, 0, 0, 2],
                    [2, 0, 0, 2],
                    [0, 2, 0, 2],
                    [2, 2, 0, 2],
                    [0, 0, 2, 2],
                    [2, 0, 2, 2],
                    [0, 2, 2, 2],
                    [2, 2, 2, 2],
                    [4, 0, 0, 4],
                    [0, 4, 0, 4],
                    [4, 4, 0, 4],
                ]
            ),
        )
        volumes = mesh.cell_volumes

        assert np.allclose(volumes, [64000] * 8 + [512000] * 3)


def test_octree(tmp_path: Path):
    name = "MyTestOctree"
    h5file_path = tmp_path / r"octree.geoh5"

    with Workspace.create(h5file_path) as workspace:
        # Create an octree mesh with variable dimensions
        with pytest.raises(TypeError, match=r"Attribute 'u_count' must"):
            Octree.create(
                workspace,
                u_count=32.0,
            )

        with pytest.raises(ValueError, match="power of 2"):
            Octree.create(
                workspace,
                u_count=15,
            )

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

        for attr in [
            "u_count",
            "v_count",
            "w_count",
        ]:
            with pytest.raises(
                AttributeError,
            ):
                setattr(mesh, attr, 12.0)

        for attr in [
            "u_cell_size",
            "v_cell_size",
            "w_cell_size",
        ]:
            with pytest.raises(
                TypeError, match=re.escape(f"Attribute '{attr}' must be type(float).")
            ):
                setattr(mesh, attr, "abc")

        assert mesh.n_cells == 8, "Number of octree cells after base_refine is wrong"

        # Refine
        workspace.save_entity(mesh)

        # Read the mesh back in
        new_workspace = Workspace(h5file_path)
        rec_obj = new_workspace.get_entity(name)[0]

        compare_entities(mesh, rec_obj)
        assert np.allclose(mesh.centroids, mesh.locations)


def test_change_octree_cells(tmp_path: Path):
    name = "MyTestOctree"
    h5file_path = tmp_path / r"octree.geoh5"

    params = {
        "origin": [0, 0, 0],
        "u_count": 32,
        "v_count": 16,
        "w_count": 8,
        "u_cell_size": 1.0,
        "v_cell_size": 1.0,
        "w_cell_size": 2.0,
        "rotation": 45,
    }
    with Workspace.create(h5file_path) as workspace:
        # Create an octree mesh with variable dimensions

        orig_mesh = Octree.create(workspace, name="original", **params)

    base_cells = orig_mesh.octree_cells

    # Refinement on first cell
    new_cells = np.vstack(
        [
            [0, 0, 0, 4],
            [4, 0, 0, 4],
            [0, 4, 0, 4],
            [4, 4, 0, 4],
        ]
    )
    octree_cells = np.vstack([new_cells, np.asarray(base_cells.tolist())[1:]])

    with workspace.open():
        mesh = Octree.create(workspace, name=name, octree_cells=octree_cells, **params)

    with workspace.open():
        rec_obj = workspace.get_entity(name)[0]
        compare_entities(mesh, rec_obj)

        rec_obj.add_data({"values": {"values": np.random.randn(rec_obj.n_cells)}})
        extent_grid = rec_obj.copy_from_extent(np.vstack([[-3, 3], [3, 9]]))
        assert (~np.isnan(extent_grid.children[0].values)).sum() == 3

    # Revert back using recarray
    with workspace.open():
        base_mesh = Octree.create(
            workspace, name="base_cells", octree_cells=base_cells, **params
        )

    with workspace.open():
        rec_obj = workspace.get_entity("base_cells")[0]
        compare_entities(base_mesh, rec_obj)

        with pytest.raises(
            ValueError, match="New octree_cells array must have the same shape"
        ):
            base_mesh.octree_cells = octree_cells
