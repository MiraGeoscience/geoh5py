#  Copyright (c) 2023 Mira Geoscience Ltd.
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
from geoh5py.shared.merging.points import PointsMerger
from geoh5py.workspace import Workspace

# import pytest


def test_merge_point_data(tmp_path):
    h5file_path = tmp_path / r"testPoints.geoh5"
    points = []
    data = []
    with Workspace.create(h5file_path) as workspace:
        points.append(
            Points.create(workspace, vertices=np.random.randn(10, 3), allow_move=False)
        )

        data.append(
            points[0].add_data(
                {
                    "DataValues0": {
                        "association": "VERTEX",
                        "values": np.random.randn(10),
                    }
                }
            )
        )

        data.append(
            points[0].add_data(
                {
                    "DataValues1": {
                        "association": "VERTEX",
                        "values": np.random.randn(10),
                    }
                }
            )
        )

        data.append(
            points[0].add_data(
                {
                    "DataValues2": {
                        "association": "VERTEX",
                        "values": np.random.randn(10),
                    }
                }
            )
        )

        entity_type = data[0].entity_type

        points.append(
            Points.create(workspace, vertices=np.random.randn(10, 3), allow_move=False)
        )
        data.append(
            points[1].add_data(
                {
                    "DataValues3": {
                        "association": "VERTEX",
                        "values": np.random.randn(10),
                        "entity_type": entity_type,
                    }
                }
            )
        )
        data.append(
            points[1].add_data(
                {
                    "DataValues4": {
                        "association": "VERTEX",
                        "values": np.random.randn(10),
                    }
                }
            )
        )

        test = PointsMerger.merge_objects(points)

        print(test)
