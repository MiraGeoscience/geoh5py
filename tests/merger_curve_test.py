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

from geoh5py.objects import Curve
from geoh5py.shared.merging import CurveMerger
from geoh5py.workspace import Workspace


# import pytest


def test_merge_curve(tmp_path):
    h5file_path = tmp_path / r"testCurve.geoh5"
    curves = []
    data = []
    with Workspace.create(h5file_path) as workspace:
        curves.append(
            Curve.create(
                workspace,
                name="points1",
                vertices=np.random.randn(10, 3),
                parts=[0, 0, 0, 0, 1, 1, 1, 2, 2, 2],
                allow_move=False,
            )
        )

        data.append(
            curves[0].add_data(
                {
                    "DataValues0": {
                        "association": "VERTEX",
                        "values": np.random.randn(10),
                    }
                }
            )
        )

        data.append(
            curves[0].add_data(
                {
                    "DataValues1": {
                        "association": "VERTEX",
                        "values": np.random.randn(10),
                    }
                }
            )
        )

        curves[0].add_data(
            {
                "TestText": {
                    "type": "text",
                    "values": np.asarray(["values"]),
                }
            }
        )

        entity_type = data[0].entity_type

        curves.append(
            Curve.create(
                workspace,
                name="points2",
                vertices=np.random.randn(10, 3),
                parts=[0, 0, 0, 0, 1, 1, 1, 2, 2, 2],
                allow_move=False,
            )
        )
        data.append(
            curves[1].add_data(
                {
                    "DataValues0": {
                        "association": "VERTEX",
                        "values": np.random.randn(10),
                        "entity_type": entity_type,
                    }
                }
            )
        )
        data.append(
            curves[1].add_data(
                {
                    "DataValues3": {
                        "association": "VERTEX",
                        "values": np.random.randn(10),
                    }
                }
            )
        )

        merged_curve = CurveMerger.merge_objects(workspace, curves)

        assert (
            merged_curve.parts
            == [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5]
        ).all()

        assert (
            merged_curve.vertices == np.vstack((curves[0].vertices, curves[1].vertices))
        ).all()

        with pytest.raises(TypeError, match="The input entities must be a list of"):
            CurveMerger.validate_type("bidon")
