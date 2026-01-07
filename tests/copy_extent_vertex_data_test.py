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

import numpy as np

from geoh5py.groups import ContainerGroup
from geoh5py.objects import Curve, Points, Surface
from geoh5py.workspace import Workspace


def test_clip_point_data(tmp_path):
    # Generate a random cloud of points
    values = np.random.randn(100)
    vertices = np.random.randn(100, 3)
    extent = np.c_[
        np.percentile(vertices, 25, axis=0), np.percentile(vertices, 75, axis=0)
    ].T

    np.savetxt(tmp_path / r"numpy_array.txt", vertices)

    clippings = np.all(
        np.c_[
            np.all(vertices >= extent[0, :], axis=1),
            np.all(vertices <= extent[1, :], axis=1),
        ],
        axis=1,
    )
    h5file_path = tmp_path / r"testClipPoints.geoh5"
    with Workspace.create(h5file_path) as workspace:
        points = Points.create(workspace, vertices=vertices, allow_move=False)
        data = points.add_data(
            {
                "DataValues": {"association": "VERTEX", "values": values},
                "TextValues": {"association": "OBJECT", "values": "abc"},
            }
        )
        points.add_file(tmp_path / "numpy_array.txt")

        with Workspace.create(tmp_path / r"testClipPoints_copy.geoh5") as new_workspace:
            clipped_pts = points.copy_from_extent(parent=new_workspace, extent=extent)
            clipped_d = clipped_pts.get_data("DataValues")[0]
            assert clipped_pts.n_vertices == clippings.sum()
            assert np.all(clipped_d.values == data[0].values[clippings])


def test_clip_curve_data(tmp_path):
    # Generate a random cloud of points
    x_loc = np.arange(0, 100)
    y_loc = np.random.randn(100)
    z_loc = np.random.randn(100)
    vertices = np.c_[x_loc, y_loc, z_loc]
    extent = np.vstack([[9.5, -100, -100], [97.5, 100, 100]])
    clippings = np.all(
        np.c_[
            np.all(vertices >= extent[0, :], axis=1),
            np.all(vertices <= extent[1, :], axis=1),
        ],
        axis=1,
    )

    h5file_path = tmp_path / r"testClipCurve.geoh5"
    with Workspace.create(h5file_path) as workspace:
        curve = Curve.create(workspace, vertices=vertices, allow_move=False)
        data = curve.add_data(
            {
                "VertexValues": {
                    "association": "VERTEX",
                    "values": np.random.randn(curve.n_vertices),
                },
                "CellValues": {
                    "association": "CELL",
                    "values": np.random.randn(curve.n_cells),
                },
                "ObjectValues": {
                    "association": "OBJECT",
                    "values": np.random.randn(1000),
                },
            }
        )
        with Workspace.create(tmp_path / r"testClipCurve_copy.geoh5") as new_workspace:
            clipped_pts = curve.copy_from_extent(parent=new_workspace, extent=extent)
            assert (
                len(clipped_pts.get_data("VertexValues")[0].values) == clippings.sum()
            )
            assert np.all(
                clipped_pts.get_data("VertexValues")[0].values
                == data[0].values[clippings]
            )
            assert (
                len(clipped_pts.get_data("CellValues")[0].values) == clipped_pts.n_cells
            )

            clipping_inverse = curve.copy_from_extent(
                parent=new_workspace, extent=extent, inverse=True
            )
            assert clipping_inverse.n_vertices == curve.n_vertices - clippings.sum()
            assert clipping_inverse.n_cells == 10
            assert np.all(
                clipping_inverse.get_data("CellValues")[0].values
                == np.r_[data[1].values[0:9], data[1].values[-1]]
            )

            np.testing.assert_allclose(
                data[2].values, clipping_inverse.get_data("ObjectValues")[0].values
            )

    # Repeat with 2D bounds - single point left
    extent = np.vstack([[-1, -100], [0.5, 100]])
    with workspace.open():
        with Workspace.create(
            tmp_path / r"testClipPoints_copy2D.geoh5"
        ) as new_workspace:
            clipped_pts = curve.copy_from_extent(parent=new_workspace, extent=extent)
            assert clipped_pts is None


def test_clip_surface(tmp_path):
    vertices = np.vstack(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    cells = np.vstack(
        [
            [0, 1, 2],
            [1, 3, 2],
        ]
    )
    h5file_path = tmp_path / r"testClipSurface.geoh5"
    with Workspace.create(h5file_path) as workspace:
        surf = Surface.create(workspace, vertices=vertices, cells=cells)

        # Clip center point, no cells left
        surf_copy = surf.copy_from_extent(
            parent=workspace, extent=np.array([[0.5, -0.5], [1.5, 0.5]]), inverse=True
        )

        assert surf_copy is None

        # Clip one point of triangle, one cell left
        surf_copy = surf.copy_from_extent(
            parent=workspace, extent=np.array([[-0.5, -0.5], [0.5, 0.5]]), inverse=True
        )

        assert len(surf_copy.cells) == 1
        assert np.sum(surf_copy.vertices - vertices[1:, :]) == 0

        # Keep three points of triangle, one cell left
        surf_copy = surf.copy_from_extent(
            parent=workspace, extent=np.array([[-0.5, -0.5], [1.5, 1.5]])
        )

        assert len(surf_copy.cells) == 1

        # Keep two points of triangle, no object
        surf_copy = surf.copy_from_extent(
            parent=workspace, extent=np.array([[0.5, -0.5], [1.5, 1.5]])
        )

        assert surf_copy is None


def test_clip_groups(tmp_path):
    vertices = np.random.randn(100, 3)
    extent = np.c_[
        np.percentile(vertices, 10, axis=0), np.percentile(vertices, 90, axis=0)
    ].T
    h5file_path = tmp_path / r"testClipGroup.geoh5"
    with Workspace.create(h5file_path) as workspace:
        group_a = ContainerGroup.create(workspace, name="Group A")
        group_b = ContainerGroup.create(workspace, name="Group B", parent=group_a)
        curve_a = Curve.create(workspace, vertices=vertices, parent=group_a)
        curve_b = Curve.create(
            workspace,
            vertices=((1000.0, 1000.0, 0.0), (1001.0, 1000.0, 0.0)),
            parent=group_b,
        )
        curve_a.add_data(
            {
                "values_a": {
                    "values": np.random.randn(curve_a.n_vertices),
                },
            }
        )
        curve_b.add_data(
            {
                "values_a": {
                    "values": np.random.randn(curve_b.n_vertices),
                },
            }
        )

        with Workspace.create(tmp_path / r"testClipPoints_copy.geoh5") as new_workspace:
            group_a.copy_from_extent(
                parent=new_workspace, clear_cache=True, extent=extent
            )

            assert len(new_workspace.objects) == 1, (
                "Error removing curve without nodes."
            )
            assert len(new_workspace.groups) == 2, "Error removing empty group."
            assert (
                getattr(new_workspace.groups[0].children[0], "_vertices", None) is None
            )
