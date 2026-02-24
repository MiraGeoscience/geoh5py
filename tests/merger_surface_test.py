# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2020-2026 Mira Geoscience Ltd.                                     '
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
import pytest

from geoh5py.objects import Surface
from geoh5py.shared.merging import SurfaceMerger
from geoh5py.workspace import Workspace


def test_merge_surface(tmp_path):
    h5file_path = tmp_path / r"testSurface.geoh5"

    surfaces = []
    with Workspace.create(h5file_path) as workspace:
        # Create a grid of points and triangulate
        x, y = np.meshgrid(np.arange(10), np.arange(10))
        x, y = x.ravel(), y.ravel()
        z = np.random.randn(x.shape[0])
        xyz = np.c_[x, y, z]

        simplices = np.unique(
            np.random.randint(0, xyz.shape[0] - 1, (xyz.shape[0], 3)), axis=1
        )

        surface0 = Surface.create(
            workspace, name="mySurf0", vertices=xyz, cells=simplices.tolist()
        )
        surfaces.append(surface0)

        z = np.random.randn(x.shape[0]) + 5
        xyz = np.c_[x, y, z]

        surface1 = Surface.create(
            workspace, name="mySurf1", vertices=xyz, cells=simplices.tolist()
        )

        surfaces.append(surface1)

        merged_surface = SurfaceMerger.merge_objects(workspace, surfaces)

        assert np.all(
            merged_surface.vertices == np.vstack([surface0.vertices, surface1.vertices])
        )

        assert np.all(
            merged_surface.cells
            == np.vstack([surface0.cells, surface1.cells + np.max(simplices) + 1])
        )

        with pytest.raises(TypeError, match="The input entities must be a list of"):
            SurfaceMerger.validate_type("bidon")
