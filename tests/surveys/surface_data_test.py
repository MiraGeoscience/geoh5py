# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2020-2026 Mira Geoscience Ltd.                                '
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

from pathlib import Path

import numpy as np
import pytest

from geoh5py.objects import Surface
from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace


def generate_surface(n_points):
    x, y = np.meshgrid(np.arange(n_points), np.arange(n_points))
    x, y = x.ravel(), y.ravel()
    z = np.random.randn(x.shape[0])

    vertices = np.c_[x, y, z]

    cells = np.unique(
        np.random.randint(0, vertices.shape[0] - 1, (vertices.shape[0], 3)), axis=1
    )

    # Create random data
    values = np.mean(np.c_[x[cells[:, 0]], x[cells[:, 1]], x[cells[:, 2]]], axis=1)

    return vertices, cells, values


def test_create_surface_data(tmp_path: Path):
    h5file_path = tmp_path / r"testSurface.geoh5"

    with Workspace.create(h5file_path) as workspace:
        # Create a grid of points and triangulate
        xyz, simplices, values = generate_surface(10)

        with pytest.warns(UserWarning, match="Attribute 'vertices' has fewer elements"):
            singleton = Surface.create(
                workspace,
            )
        assert singleton.n_vertices == 3, "Error creating surface with no vertices."
        assert singleton.n_cells == 1, "Error creating surface with no cells."

        # Create a geoh5 surface
        with pytest.raises(ValueError, match="Array of 'cells' should be of shape"):
            Surface.create(workspace, name="mySurf", vertices=xyz, cells=np.c_[[0, 1]])

        with pytest.raises(TypeError, match="Indices array must be of integer type"):
            Surface.create(
                workspace, name="mySurf", vertices=xyz, cells=simplices.astype(float)
            )

        surface = Surface.create(
            workspace, name="mySurf", vertices=xyz, cells=simplices.tolist()
        )

        data = surface.add_data({"TMI": {"values": values}})

        # Read the object from a different workspace object on the same file
        new_workspace = Workspace(h5file_path)

        rec_obj = new_workspace.get_entity("mySurf")[0]
        rec_data = rec_obj.get_data("TMI")[0]

        compare_entities(surface, rec_obj)
        compare_entities(data, rec_data)


def test_remove_cells_surface_data(tmp_path: Path):
    xyz, simplices, values = generate_surface(10)

    with Workspace.create(tmp_path / "remove_cells.geoh5") as workspace:
        surface = Surface.create(
            workspace, name="mySurf", vertices=xyz, cells=simplices.tolist()
        )

        surface.add_data({"TMI": {"values": values}})

        with pytest.raises(
            ValueError, match="Found indices larger than the number of cells."
        ):
            surface.remove_cells([101])

        with pytest.raises(
            ValueError, match="New cells array must have the same shape"
        ):
            surface.cells = surface.cells[1:, :]

        surface.remove_cells([0])

        assert len(surface.children[0].values) == 99, (
            "Error removing data values with cells."
        )


def test_remove_vertices_surface_data(tmp_path: Path):
    with Workspace.create(tmp_path / "remove_vertices.geoh5") as workspace:
        xyz, simplices, values = generate_surface(10)

        surface = Surface.create(
            workspace, name="mySurf", vertices=xyz, cells=simplices.tolist()
        )

        surface.add_data({"TMI": {"values": values}})

        np.random.seed(42)

        data = surface.add_data(
            {
                "cellValues": {
                    "values": np.random.randn(surface.n_cells).astype(np.float64)
                },
            }
        )

        with pytest.raises(
            ValueError, match="Found indices larger than the number of vertices."
        ):
            surface.remove_vertices([1001])

        logic = np.ones(surface.n_vertices)
        logic[[0, 3]] = False
        expected = np.all(logic[surface.cells], axis=1).sum()

        surface.remove_vertices([0, 3])

        assert len(data.values) == expected, "Error removing data values with cells."
        assert len(surface.vertices) == 98, "Error removing vertices from cells."
