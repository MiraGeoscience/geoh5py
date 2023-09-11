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
import pytest

from geoh5py.data import Data
from geoh5py.groups import PropertyGroup
from geoh5py.objects import Points, Surface
from geoh5py.shared.merging import PointsMerger
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

        test = PropertyGroup(parent=points[0], name="test")

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
                    "DataValues2": {
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
        merged_data = list(
            dict(
                sorted(
                    {
                        child.name: child.values
                        for child in test.children
                        if isinstance(child, Data)
                    }.items()
                )
            ).values()
        )

        np.testing.assert_almost_equal(
            merged_data[0], np.hstack((data[0].values, data[2].values))
        )

        np.testing.assert_almost_equal(
            merged_data[1], np.hstack((data[1].values, nan_array))
        )

        np.testing.assert_almost_equal(
            merged_data[2], np.hstack((nan_array, data[3].values))
        )


def test_merge_point_data_unique_entity_name(tmp_path):
    """
    Test the suboptimal scenario where all the data pairs of entity type
        and name is unique in the objects.
    """
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
                    "DataValues": {
                        "association": "VERTEX",
                        "values": np.random.randn(10),
                    }
                }
            )
        )

        entity_type = data[0].entity_type

        data.append(
            points[0].add_data(
                {
                    "DataValues1": {
                        "association": "VERTEX",
                        "values": np.random.randn(10),
                        "entity_type": entity_type,
                    }
                }
            )
        )

        points.append(
            Points.create(workspace, vertices=np.random.randn(10, 3), allow_move=False)
        )

        data.append(
            points[1].add_data(
                {
                    "DataValues": {
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
        merged_data = list(
            dict(
                sorted(
                    {
                        child.name: child.values
                        for child in test.children
                        if isinstance(child, Data)
                    }.items()
                )
            ).values()
        )

        np.testing.assert_almost_equal(
            merged_data[1], np.hstack((data[0].values, data[2].values))
        )

        np.testing.assert_almost_equal(
            merged_data[2], np.hstack((data[1].values, nan_array))
        )

        np.testing.assert_almost_equal(
            merged_data[0], np.hstack((nan_array, data[3].values))
        )


def test_merge_point_data_unique_entity_name_unique_name(tmp_path):
    """
    Test the wost case scenario where data inside objects are not uniques.
    """
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
                    "DataValues": {
                        "association": "VERTEX",
                        "values": np.random.randn(10),
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
                        "values": np.random.randn(10),
                        "entity_type": entity_type,
                    }
                }
            )
        )

        points.append(
            Points.create(workspace, vertices=np.random.randn(10, 3), allow_move=False)
        )

        data.append(
            points[1].add_data(
                {
                    "DataValues": {
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
        merged_data = list(
            dict(
                sorted(
                    {
                        child.name: child.values
                        for child in test.children
                        if isinstance(child, Data)
                    }.items()
                )
            ).values()
        )

        np.testing.assert_almost_equal(
            merged_data[0], np.hstack((nan_array, data[3].values))
        )

        np.testing.assert_almost_equal(
            merged_data[1], np.hstack((data[0].values, nan_array))
        )

        np.testing.assert_almost_equal(
            merged_data[2], np.hstack((data[1].values, nan_array))
        )

        np.testing.assert_almost_equal(
            merged_data[3], np.hstack((nan_array, data[2].values))
        )


def test_merge_attribute_error(tmp_path):
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
                    "DataValues": {
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
                    "DataValues": {
                        "association": "CELL",
                        "values": np.random.randn(10),
                        "entity_type": entity_type,
                    }
                }
            )
        )

        with pytest.raises(
            ValueError, match="Cannot merge data with different associations"
        ):
            _ = PointsMerger.merge_objects(workspace, points)

        with pytest.raises(TypeError, match="The input entities must be a list"):
            _ = PointsMerger.merge_objects(workspace, "bidon")

        with pytest.raises(ValueError, match="Need more than one object"):
            _ = PointsMerger.merge_objects(workspace, [points[0]])

        surface = Surface(
            workspace,
            vertices=np.random.randn(10, 3),
        )

        with pytest.raises(TypeError, match="All objects must be of"):
            _ = PointsMerger.merge_objects(workspace, [points[0], surface])

        surface2 = Surface(
            workspace,
            vertices=np.random.randn(10, 3),
        )

        with pytest.raises(TypeError, match="The input entities must be a list"):
            _ = PointsMerger.merge_objects(workspace, [surface, surface2])

        points[0] = Points(workspace)

        with pytest.raises(AttributeError, match="All entities must have vertices"):
            _ = PointsMerger.merge_objects(workspace, points)

        point = Points.create(
            workspace,
            vertices=np.random.randn(10, 3),
            allow_move=False,
            name="visual parameter",
        )

        point.add_data(
            {
                "Visual Parameters": {
                    "association": "VERTEX",
                    "values": np.random.randn(10),
                }
            }
        )

        res = PointsMerger.extract_data_information([point])

        assert (res[0] == np.array([], dtype=object)).all()
        assert res[1:] == ([], 10, 1)
