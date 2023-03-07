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

from geoh5py.io import H5Writer
from geoh5py.objects import Points
from geoh5py.shared import fetch_h5_handle
from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace


def test_create_point_data(tmp_path):
    new_name = "TestName"
    # Generate a random cloud of points
    values = np.random.randn(12)
    h5file_path = tmp_path / r"testPoints.geoh5"
    workspace = Workspace(h5file_path)
    points = Points.create(workspace, vertices=np.random.randn(12, 3), allow_move=False)
    data = points.add_data({"DataValues": {"association": "VERTEX", "values": values}})

    with pytest.raises(ValueError, match="Association flag should be one of"):
        points.add_data({"test": {"association": "ABC", "values": values}})

    with pytest.raises(TypeError, match="Association must be of type"):
        points.add_data({"test": {"association": Points, "values": values}})

    tag = points.add_data(
        {"my_comment": {"association": "OBJECT", "values": "hello_world"}}
    )
    # Change some data attributes for testing
    data.allow_delete = False
    data.allow_move = True
    data.allow_rename = False
    data.name = new_name
    # Fake ANALYST creating a StatsCache
    with fetch_h5_handle(h5file_path, mode="r+") as h5file:
        etype_handle = H5Writer.fetch_handle(h5file, data.entity_type)
        etype_handle.create_group("StatsCache")
    # Trigger replace of values
    data.values = values * 2.0

    # Read the data back in from a fresh workspace
    new_workspace = Workspace(h5file_path)
    rec_obj = new_workspace.get_entity("Points")[0]
    rec_data = new_workspace.get_entity(new_name)[0]
    rec_tag = new_workspace.get_entity("my_comment")[0]
    compare_entities(points, rec_obj)
    compare_entities(data, rec_data)
    compare_entities(tag, rec_tag)
    with fetch_h5_handle(h5file_path) as h5file:
        etype_handle = H5Writer.fetch_handle(h5file, rec_data.entity_type)
        assert (
            etype_handle.get("StatsCache") is None
        ), "StatsCache was not properly deleted on update of values"


def test_remove_point_data(tmp_path):
    # Generate a random cloud of points
    values = np.random.randn(12)
    h5file_path = tmp_path / r"testPoints.geoh5"
    with Workspace(h5file_path) as workspace:
        points = Points.create(workspace)

        with pytest.warns(UserWarning, match="No vertices to be removed."):
            points.remove_vertices([12])

        with pytest.raises(ValueError, match="Array of vertices must be of shape"):
            points.vertices = np.r_[1, 2, 3]

        points.vertices = np.random.randn(12, 3)

        assert (
            points.mask_by_extent([[1000, 1000], [1001, 1001]]) is None
        ), "Error returning None mask."

        with pytest.raises(TypeError, match="Indices must be a list or numpy array."):
            points.remove_vertices("abc")

        data = points.add_data(
            {"DataValues": {"association": "VERTEX", "values": values}}
        )

        with pytest.raises(
            ValueError, match="Attempting to assign 'vertices' with fewer values."
        ):
            points.vertices = np.random.randn(10, 3)

        with pytest.raises(
            ValueError, match="Found indices larger than the number of vertices."
        ):
            points.remove_vertices([12])

        points.remove_vertices([1, 2])

        assert len(data.values) == 10, "Error removing data values with vertices."

        assert (
            points.mask_by_extent([[1e6, 1e6], [2e6, 2e6]]) is None
        ), "Error masking points by extent."


def test_copy_points_data(tmp_path):
    values = np.random.randn(12)
    h5file_path = tmp_path / r"testPoints.geoh5"
    with Workspace(h5file_path) as workspace:
        points = Points.create(workspace)
        points.vertices = np.random.randn(12, 3)
        data = points.add_data(
            {"DataValues": {"association": "VERTEX", "values": values}}
        )

        with pytest.raises(ValueError, match="Mask must be an array of shape"):
            points.copy(mask=np.r_[1, 2, 3])

        with pytest.raises(ValueError, match="Mask must be a boolean array of shape"):
            data.copy(mask=np.r_[1, 2, 3])

        with pytest.raises(TypeError, match="Mask must be an array or None."):
            data.copy(mask="abc")

        mask = np.zeros(12, dtype=bool)
        mask[:4] = True
        copy_data = data.copy(mask=mask)

        assert np.isnan(copy_data.values).sum() == 8, "Error copying data."

        ind = np.all(points.vertices[:, :2] > 0, axis=1) & np.all(
            points.vertices[:, :2] < 2, axis=1
        )
        mask = data.mask_by_extent([[0, 0], [2, 2]])

        assert np.all(mask == ind), "Error masking data by extent."
