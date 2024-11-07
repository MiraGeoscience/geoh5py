#  Copyright (c) 2024 Mira Geoscience Ltd.
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

import random
import string

import numpy as np
import pytest

from geoh5py.data import GeometricDataConstants, ReferencedData, ReferenceValueMap
from geoh5py.data.data_type import ReferencedValueMapType
from geoh5py.objects import Points
from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace


def generate_value_map(workspace, n_data=12, n_class=8):
    values = np.random.randint(1, high=n_class, size=n_data)
    refs = np.unique(values)
    value_map = {}
    for ref in refs:
        value_map[ref] = "".join(
            random.choice(string.ascii_lowercase) for i in range(n_class)
        )

    points = Points.create(
        workspace, vertices=np.random.randn(n_data, 3), allow_move=False
    )

    data = points.add_data(
        {
            "DataValues": {
                "type": "referenced",
                "values": values,
                "value_map": value_map,
            }
        }
    )
    data.entity_type.name = "abc"
    return points, data


def test_reference_value_map():
    workspace = Workspace()

    with pytest.raises(TypeError, match="Value map must be a numpy array or dict."):
        ReferenceValueMap("value_map")

    with pytest.raises(KeyError, match="Key must be an positive integer"):
        ReferenceValueMap({-1: "test"})

    with pytest.raises(ValueError, match="Value for key 0 must be b'Unknown'"):
        ReferencedValueMapType(workspace, value_map=((0, "test"),))

    with pytest.raises(ValueError, match="Array of 'value_map' must be of dtype"):
        array = np.array([(0, "test")], dtype=[("I", "i1"), ("K", "<U13")])
        ReferenceValueMap(array)

    random_array = np.random.randint(1, high=10, size=1000)
    value_map = ReferenceValueMap(random_array)

    assert len(value_map) == 9
    assert isinstance(value_map(), dict)


def test_create_reference_data(tmp_path):
    h5file_path = tmp_path / r"testPoints.geoh5"

    with Workspace.create(h5file_path) as workspace:
        points, data = generate_value_map(workspace)

        with pytest.raises(
            TypeError, match="entity_type must be of type ReferenceDataType"
        ):
            data.entity_type = "abc"

        new_workspace = Workspace(h5file_path)
        rec_obj = new_workspace.get_entity("Points")[0]
        rec_data = new_workspace.get_entity("DataValues")[0]

        compare_entities(points, rec_obj)
        compare_entities(data, rec_data, ignore=["_map"])

        assert data.entity_type.value_map() == rec_data.entity_type.value_map()

        assert data.mapped_values[0] == dict(data.value_map.map)[data.values[0]]

        data._entity_type._value_map = None  #  pylint: disable=protected-access

        with pytest.raises(ValueError, match="Entity type must have a value map."):
            _ = data.mapped_values


def test_add_data_map(tmp_path):
    h5file_path = tmp_path / r"testPoints.geoh5"

    with Workspace.create(h5file_path) as workspace:
        _, data = generate_value_map(workspace)

        with pytest.raises(ValueError, match="Data map must be a 2D array"):
            data.add_data_map("test", np.random.randn(12))

        with pytest.raises(
            KeyError, match="Data map keys must be a subset of the value map keys."
        ):
            data.add_data_map("test", np.c_[np.arange(12), np.random.randn(12)])

        data_map = np.c_[
            data.entity_type.value_map.map["Key"],
            np.random.randn(len(data.entity_type.value_map.map["Key"])),
        ]

        with pytest.raises(TypeError, match="Property maps must be a dictionary"):
            data.data_maps = data_map

        with pytest.raises(
            TypeError, match="Data map values must be a numpy array or dict"
        ):
            data.add_data_map("test", "abc")

        assert data.remove_data_map("DataValues") is None

        value_map = data.entity_type.value_map
        data.entity_type.value_map = None

        with pytest.raises(ValueError, match="Entity type must have a value map."):
            data.add_data_map("test", data_map)

        data.entity_type.value_map = value_map
        data.add_data_map("test", data_map)

        with pytest.raises(KeyError, match="Data map 'test' already exists."):
            data.add_data_map("test", data_map)

        data_map = np.c_[
            data.entity_type.value_map.map["Key"],
            np.random.randn(len(data.entity_type.value_map.map["Key"])),
        ]

        data.add_data_map("test2", data_map)

        assert isinstance(data.data_maps["test"], GeometricDataConstants)

    with Workspace(h5file_path) as workspace:
        rec_data = workspace.get_entity("DataValues")[0]

        assert isinstance(rec_data.data_maps["test"], GeometricDataConstants)

        rec_data.remove_data_map("test")

        assert "test" not in rec_data.data_maps
        assert rec_data.parent.get_entity("test")[0] is None

        geo_data = rec_data.data_maps["test2"]
        assert geo_data.entity_type.value_map is not None
        assert geo_data.entity_type.value_map.name == "test2"
        np.testing.assert_array_almost_equal(
            np.asarray(list(geo_data.entity_type.value_map().values()), dtype=float),
            data_map[:, 1],
        )


def test_create_bytes_reference(tmp_path):
    h5file_path = tmp_path / r"testPoints.geoh5"

    with Workspace.create(h5file_path) as workspace:
        points, data = generate_value_map(workspace)

        value_map = data.entity_type.value_map()
        for key, value in value_map.items():
            value_map[key] = value.encode()

        points.add_data(
            {
                "DataValues_bytes": {
                    "type": "referenced",
                    "values": data.values,
                    "value_map": value_map,
                }
            }
        )

    with Workspace(h5file_path) as workspace:
        data = workspace.get_entity("DataValues_bytes")[0]
        assert data.entity_type.value_map.map.dtype == np.dtype(
            [("Key", "<u4"), ("Value", "O")]
        )
