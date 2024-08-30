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

from geoh5py.data import ReferenceValueMap
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
    return points, data


def test_create_reference_data(tmp_path):
    h5file_path = tmp_path / r"testPoints.geoh5"

    with Workspace.create(h5file_path) as workspace:
        points, data = generate_value_map(workspace)

        new_workspace = Workspace(h5file_path)
        rec_obj = new_workspace.get_entity("Points")[0]
        rec_data = new_workspace.get_entity("DataValues")[0]

        compare_entities(points, rec_obj)
        compare_entities(data, rec_data, ignore=["_map"])

        assert all(data.entity_type.value_map.map == rec_data.entity_type.value_map.map)

        with pytest.raises(TypeError, match="Value map must be a numpy array or dict."):
            ReferenceValueMap("value_map")

        with pytest.raises(KeyError, match="Key must be an positive integer"):
            ReferenceValueMap({-1: "test"})

        with pytest.raises(ValueError, match="Value for key 0 must be 'Unknown'"):
            ReferenceValueMap({0: "test"})

        value_map = ReferenceValueMap({0: "Unknown", 2: "test"})

        with pytest.raises(KeyError, match="Key 'test' not found in value map."):
            value_map["test"] = "test"

        value_map[2] = 1

        assert value_map[2] == "1"

        with pytest.raises(ValueError, match="Value for key 0 must be 'Unknown'"):
            value_map[0] = "test"

        assert dict(value_map()) == {0: "Unknown", 2: "1"}


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

        data.add_data_map("test", data_map)
