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

import random
import string

import numpy as np
import pytest

from geoh5py.data import GeometricDataConstants, ReferenceValueMap
from geoh5py.data.data_type import ReferencedValueMapType
from geoh5py.groups import PropertyGroup
from geoh5py.objects import Points
from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace


def generate_value_map(workspace, n_data=12, n_class=8):
    values = np.random.randint(1, high=n_class, size=n_data)
    refs = np.unique(values)
    value_map = {}
    for ref in refs:
        random_len = random.randint(1, n_class)
        value_map[ref] = "".join(
            random.choice(string.ascii_lowercase) for i in range(random_len)
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
    return points, data, value_map


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


def test_copy_reference_data(tmp_path):
    h5file_path = tmp_path / r"testPoints.geoh5"

    with Workspace.create(h5file_path) as workspace:
        points, data, _ = generate_value_map(workspace)

        new_value_map = data.entity_type.value_map()

        new_index = np.max(list(new_value_map.keys())) + 1
        extra_name = new_value_map[new_index - 1].capitalize()
        new_value_map[new_index] = extra_name

        new_type = data.entity_type.copy(value_map=new_value_map)
        new_values = data.values.copy()
        new_values[-1] = new_index
        ref_data = points.add_data(
            {"new_values": {"values": new_values, "entity_type": new_type}}
        )

        value_map = ref_data.value_map()

        assert list(value_map.values())[-1] == extra_name + "(1)"


def test_create_reference_data(tmp_path):
    h5file_path = tmp_path / r"testPoints.geoh5"

    with Workspace.create(h5file_path) as workspace:
        points, data, _ = generate_value_map(workspace)

        with pytest.raises(
            TypeError, match="Input 'entity_type' with primitive_type 'None'"
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
    h5file_path = tmp_path / (__name__ + ".geoh5")

    with Workspace.create(h5file_path) as workspace:
        _, data, _ = generate_value_map(workspace)

        # create a property group to test GEOPY-2256 bug
        parent = data.parent
        _ = PropertyGroup(parent, properties=[data])

        data_map = np.c_[
            data.entity_type.value_map.map["Key"],
            np.random.randn(len(data.entity_type.value_map.map["Key"])),
        ]

        # Add duplicate value
        data_map[1, 1] = data_map[0, 1]
        with pytest.raises(TypeError, match="Property maps must be a dictionary"):
            data.data_maps = data_map

        with pytest.raises(TypeError, match="Value map must be a numpy array or dict."):
            data.add_data_map("test", "abc")

        assert data.remove_data_map("DataValues") is None

        value_map = data.entity_type.value_map
        data.entity_type.value_map = None

        with pytest.raises(ValueError, match="Entity type must have a value map."):
            data.add_data_map("test", data_map)

        data.entity_type.value_map = value_map
        data.add_data_map("test", data_map)

        data_map = np.c_[
            data.entity_type.value_map.map["Key"],
            np.random.randn(len(data.entity_type.value_map.map["Key"])),
        ]

        data.add_data_map("test2", data_map)

        # test duplicate
        data.add_data_map("test2", data_map, public=False)

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

        geo_data_2 = rec_data.data_maps["test2(1)"]
        assert geo_data_2.entity_type.value_map.name == "test2(1)"

        assert geo_data.public == 1
        assert geo_data_2.public == 0

        assert np.array_equal(
            np.asarray(list(geo_data.entity_type.value_map().values()), dtype=float),
            np.asarray(list(geo_data_2.entity_type.value_map().values()), dtype=float),
        )


def test_copy_data_map(tmp_path):
    h5file_path = tmp_path / (__name__ + ".geoh5")

    with Workspace.create(h5file_path) as workspace:
        _, data, _ = generate_value_map(workspace)

        data_map = np.c_[
            data.entity_type.value_map.map["Key"],
            np.random.randn(len(data.entity_type.value_map.map["Key"])),
        ]
        data.add_data_map("test2", data_map)

        data.parent.copy()
        geom_data = workspace.get_entity("test2")
        assert len(geom_data) == 2

        assert geom_data[0].parent != geom_data[1].parent

        assert np.all(
            geom_data[0].entity_type.value_map.map
            == geom_data[1].entity_type.value_map.map
        )

        # test with copying data on the same parent
        data_copy = data.copy()

        assert (
            list(data.data_maps.keys())[0]
            != list(data_copy.data_maps.keys())[0]
            == "test2(1)"
        )


def test_create_bytes_reference(tmp_path):
    h5file_path = tmp_path / (__name__ + ".geoh5")

    with Workspace.create(h5file_path) as workspace:
        points, data, _ = generate_value_map(workspace)

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


def test_value_map_from_values(tmp_path):
    h5file_path = tmp_path / (__name__ + ".geoh5")

    with Workspace.create(h5file_path) as workspace:
        points, data, _ = generate_value_map(workspace)

        new = points.add_data(
            {
                "auto_map": {
                    "type": "referenced",
                    "values": data.values,
                }
            }
        )
        assert len(new.entity_type.value_map.map) == len(np.unique(data.values)) + 1


def test_value_map_from_str_values(tmp_path):
    h5file_path = tmp_path / (__name__ + ".geoh5")

    with Workspace.create(h5file_path) as workspace:
        points, _, _ = generate_value_map(workspace, n_data=100)

        value_map = []
        for _ in range(4):
            value_map.append(
                "".join(random.choice(string.ascii_lowercase) for _ in range(8))
            )

        values = np.random.randint(1, 5, size=points.n_vertices)

        new = points.add_data(
            {
                "auto_map": {
                    "type": "referenced",
                    "values": values,
                    "value_map": np.asarray(value_map),
                }
            }
        )
        assert len(new.entity_type.value_map.map) == len(np.unique(values)) + 1


def test_variable_string_length(tmp_path):
    h5file_path = tmp_path / (__name__ + ".geoh5")

    with Workspace.create(h5file_path) as workspace:
        n_class = 10
        n_data = 12
        values = np.random.randint(1, high=n_class, size=n_data)
        refs = np.unique(values)
        value_map = {}
        for ref in refs:
            value_map[ref] = random.randint(1, 10000)

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

        np.testing.assert_allclose(
            [int(val) for val in data.value_map.map["Value"] if val != b"Unknown"],
            list(value_map.values()),
        )

        _, data, orig = generate_value_map(workspace, n_class=100)

        values = [
            val.decode() for val in data.value_map.map["Value"] if val != b"Unknown"
        ]
        assert len(set(values).difference(set(orig.values()))) == 0
