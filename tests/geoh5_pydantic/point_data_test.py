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
import pytest

from geoh5py.objects import Points
from geoh5py.workspace import Workspace
from geoh5py_pydantic.points import PointsModel


def test_create_point_data(tmp_path):
    new_name = "TestName"
    # Generate a random cloud of points
    values = np.random.randn(12, 3)

    with Workspace.create(tmp_path / f"{__name__}.geoh5") as ws:
        points = Points.create(ws, vertices=values, names=new_name)

        new_points = PointsModel.from_legacy_points(points)

        assert isinstance(new_points, PointsModel)
        assert new_points.name == new_name
