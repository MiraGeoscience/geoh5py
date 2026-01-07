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

import logging

import numpy as np

from geoh5py.objects import Points
from geoh5py.workspace import Workspace


def test_create_point_data(tmp_path, caplog):
    h5file_path = tmp_path / r"testNumeric.geoh5"
    workspace = Workspace.create(h5file_path)

    values = np.random.randn(16)
    points = Points.create(workspace, vertices=np.random.randn(12, 3), allow_move=False)

    with caplog.at_level(logging.WARNING):
        data = points.add_data(
            {
                "DataValues1": {"association": "VERTEX", "values": values},
                "DataValues2": {"association": "VERTEX", "values": values[:8]},
            }
        )

    assert np.array_equal(data[0].values, values[:12])

    test = np.full((12,), np.nan)
    test[:8] = values[:8]

    assert np.array_equal(data[1].values, test, equal_nan=True)
