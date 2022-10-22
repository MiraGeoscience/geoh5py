#  Copyright (c) 2022 Mira Geoscience Ltd.
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


from __future__ import annotations

import numpy as np

from geoh5py.groups import ContainerGroup
from geoh5py.objects import Curve, Points
from geoh5py.workspace import Workspace


def test_clip_point_data(tmp_path):

    # Generate a random cloud of points
    values = np.random.randn(100)
    vertices = np.random.randn(100, 3)
    extent = np.c_[
        np.percentile(vertices, 25, axis=0), np.percentile(vertices, 75, axis=0)
    ].T

    clippings = np.all(
        np.c_[
            np.all(vertices >= extent[0, :], axis=1),
            np.all(vertices <= extent[1, :], axis=1),
        ],
        axis=1,
    )
    h5file_path = tmp_path / r"testClipPoints.geoh5"
    with Workspace(h5file_path) as workspace:
        points = Points.create(workspace, vertices=vertices, allow_move=False)
        data = points.add_data(
            {"DataValues": {"association": "VERTEX", "values": values}}
        )
        with Workspace(tmp_path / r"testClipPoints_copy.geoh5") as new_workspace:
            clipped_pts = points.copy_from_extent(extent, parent=new_workspace)
            clipped_d = clipped_pts.get_data("DataValues")[0]
            assert clipped_pts.n_vertices == clippings.sum()
            assert np.all(clipped_d.values == data.values[clippings])


def test_clip_curve_data(tmp_path):

    # Generate a random cloud of points
    vertices = np.random.randn(100, 3)
    extent = np.c_[
        np.percentile(vertices, 10, axis=0), np.percentile(vertices, 90, axis=0)
    ].T

    clippings = np.all(
        np.c_[
            np.all(vertices >= extent[0, :], axis=1),
            np.all(vertices <= extent[1, :], axis=1),
        ],
        axis=1,
    )
    h5file_path = tmp_path / r"testClipCurve.geoh5"
    with Workspace(h5file_path) as workspace:
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
            }
        )
        with Workspace(tmp_path / r"testClipPoints_copy.geoh5") as new_workspace:
            clipped_pts = curve.copy_from_extent(extent, parent=new_workspace)
            clipped_d = clipped_pts.get_data("VertexValues")[0]
            clipped_c = clipped_pts.get_data("CellValues")[0]
            assert clipped_pts.n_vertices == clippings.sum()
            assert np.all(clipped_d.values == data[0].values[clippings])
            assert len(clipped_c.values) == clipped_pts.n_cells


def test_clip_groups(tmp_path):
    vertices = np.random.randn(100, 3)
    extent = np.c_[
        np.percentile(vertices, 10, axis=0), np.percentile(vertices, 90, axis=0)
    ].T
    h5file_path = tmp_path / r"testClipGroup.geoh5"
    with Workspace(h5file_path) as workspace:
        group_a = ContainerGroup.create(workspace, name="GroupA")
        group_b = ContainerGroup.create(workspace, name="GroupB", parent=group_a)
        curve_a = Curve.create(workspace, vertices=vertices, parent=group_a)
        curve_b = Curve.create(
            workspace, vertices=np.c_[1000.0, 1000.0, 0.0], parent=group_b
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
        with Workspace(tmp_path / r"testClipPoints_copy.geoh5") as new_workspace:
            group_a.copy_from_extent(extent, parent=new_workspace)

            assert (
                len(new_workspace.objects) == 1
            ), "Error removing curve without nodes."
            assert len(new_workspace.groups) == 2, "Error removing empty group."
