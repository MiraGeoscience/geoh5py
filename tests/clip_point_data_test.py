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

from geoh5py.objects import Points
from geoh5py.workspace import Workspace


def test_clip_point_data(tmp_path):

    # Generate a random cloud of points
    values = np.random.randn(100)
    vertices = np.random.randn(100, 3)
    extent = np.c_[
        np.percentile(vertices, 25, axis=0), np.percentile(vertices, 75, axis=0)
    ].T

    h5file_path = tmp_path / r"testClipPoints.geoh5"
    with Workspace(h5file_path) as workspace:
        points = Points.create(workspace, vertices=vertices, allow_move=False)
        data = points.add_data(
            {"DataValues": {"association": "VERTEX", "values": values}}
        )
        with Workspace(tmp_path / r"testClipPoints_copy.geoh5") as new_workspace:
            points.clip_by_extent(extent, parent=new_workspace)

    print(data)
