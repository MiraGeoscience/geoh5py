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

from geoh5py.groups import PropertyGroup
from geoh5py.objects import Points, Surface
from geoh5py.shared.merging import PointsMerger
from geoh5py.shared.merging.base import BaseMerger
from geoh5py.workspace import Workspace


def test_merge_point_data_unique_entity(tmp_path):
    """
    Test the optimal scenario where all the data entity type is unique in the objects.
    """
    h5file_path = tmp_path / r"testPoints.geoh5"
    points = []
    data = []
    with Workspace.create(h5file_path) as workspace:
        points.append(
            Points.create(
                workspace,
                name="points1",
                vertices=np.random.randn(10, 3),
                allow_move=False,
            )
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
        points[0].add_data(
            {
                "TestText": {
                    "type": "text",
                    "values": np.asarray(["values"]),
                }
            }
        )

        entity_type = data[0].entity_type

        points.append(
            Points.create(
                workspace,
                name="points2",
                vertices=np.random.randn(10, 3),
                allow_move=False,
            )
        )
        data.append(
            points[1].add_data(
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
            points[1].add_data(
                {
                    "DataValues3": {
                        "association": "VERTEX",
                        "values": np.random.randn(10),
                    }
                }
            )
        )

        test = PointsMerger.merge_objects(workspace, points)

        nan_array = np.empty(10)
        nan_array[:] = np.nan

        # sort the dictionary by its keys
        np.testing.assert_almost_equal(
            test.children[0].values, np.hstack((data[0].values, data[2].values))
        )

        np.testing.assert_almost_equal(
            test.children[1].values, np.hstack((data[1].values, nan_array))
        )

        np.testing.assert_almost_equal(
            test.children[2].values, np.hstack((nan_array, data[3].values))
        )


def test_merge_point_data_unique_entity_name_unique_name(tmp_path):
    """
    Test the wost case scenario where data inside objects are not uniques.
    """
    h5file_path = tmp_path / r"testPoints.geoh5"
    points = []
    data = []
    with Workspace.create(h5file_path) as workspace_init:
        points.append(
            Points.create(
                workspace_init, vertices=np.random.randn(10, 3), allow_move=False
            )
        )

        data.append(
            points[0].add_data(
                {
                    "DataValues": {
                        "association": "VERTEX",
                        "values": np.random.randint(10, size=10, dtype=np.int32),
                    }
                }
            )
        )

        entity_type = data[0].entity_type

        data.append(
            points[0].add_data(
                {
                    "DataValues": {
                        "association": "VERTEX",
                        "values": np.random.randint(10, size=10, dtype=np.int32),
                        "entity_type": entity_type,
                    }
                }
            )
        )

        points.append(
            Points.create(
                workspace_init, vertices=np.random.randn(10, 3), allow_move=False
            )
        )

        data.append(
            points[1].add_data(
                {
                    "DataValues": {
                        "association": "VERTEX",
                        "values": np.random.randint(10, size=10, dtype=np.int32),
                        "entity_type": entity_type,
                    }
                }
            )
        )
        data.append(
            points[1].add_data(
                {
                    "DataValues3": {
                        "association": "VERTEX",
                        "values": np.random.randn(10),
                    }
                }
            )
        )

    h5file_path_2 = tmp_path / r"testPoints2.geoh5"
    with Workspace.create(h5file_path_2) as workspace:
        test = PointsMerger.merge_objects(workspace, points)

        int_nan_array = np.empty(10)
        int_nan_array[:] = data[0].nan_value

        float_nan_array = np.empty(10)
        float_nan_array[:] = np.nan

        np.testing.assert_almost_equal(
            test.children[0].values, np.hstack((data[0].values, data[2].values))
        )

        np.testing.assert_almost_equal(
            test.children[1].values, np.hstack((data[1].values, int_nan_array))
        )

        np.testing.assert_almost_equal(
            test.children[2].values, np.hstack((float_nan_array, data[3].values))
        )


def test_merge_attribute_error(tmp_path):
    h5file_path = tmp_path / r"testPoints.geoh5"
    points = []
    data = []
    with Workspace.create(h5file_path) as workspace_init:
        points.append(
            Points.create(
                workspace_init, vertices=np.random.randn(10, 3), allow_move=False
            )
        )

        data.append(
            points[0].add_data(
                {
                    "DataValues": {
                        "association": "VERTEX",
                        "values": np.random.randn(10),
                    }
                }
            )
        )

        entity_type = data[0].entity_type

        points.append(
            Points.create(
                workspace_init, vertices=np.random.randn(10, 3), allow_move=False
            )
        )

        data.append(
            points[1].add_data(
                {
                    "DataValues": {
                        "association": "CELL",
                        "values": np.random.randn(10),
                        "entity_type": entity_type,
                    }
                }
            )
        )

    h5file_path_2 = tmp_path / r"testPoints2.geoh5"
    with Workspace.create(h5file_path_2) as workspace:
        with pytest.raises(TypeError, match="The input entities must be a list"):
            _ = PointsMerger.merge_objects(workspace, "bidon")

        with pytest.raises(ValueError, match="Need more than one object"):
            _ = PointsMerger.merge_objects(workspace, [points[0]])

        surface = Surface.create(
            workspace,
            vertices=np.random.randn(10, 3),
        )

        with pytest.raises(TypeError, match="All objects must be of"):
            _ = PointsMerger.merge_objects(workspace, [points[0], surface])

        surface2 = Surface.create(
            workspace,
            vertices=np.random.randn(10, 3),
        )

        with pytest.raises(TypeError, match="The input entities must be a list"):
            _ = PointsMerger.merge_objects(workspace, [surface, surface2])

        points[0] = Points.create(workspace)

        with pytest.raises(NotImplementedError, match="BaseMerger cannot be use"):
            _ = BaseMerger.create_object(workspace, points, name="bidon")

        with pytest.raises(NotImplementedError, match="BaseMerger cannot be use"):
            _ = BaseMerger.validate_structure(points[0])
