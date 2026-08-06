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

import pickle
from uuid import UUID, uuid4

import numpy as np
import pytest
from pydantic import ValidationError

from geoh5py.objects import Points
from geoh5py.workspace import Workspace
from geoh5py_pydantic import (
    VERTICES_DTYPE,
    CallableArraySource,
    Geoh5EntityPayload,
    LazyArray,
    PointsModel,
)


def test_points_model_is_workspace_free_and_picklable(tmp_path):
    """Points can be used and persisted without creating a Workspace."""
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    points = PointsModel(name="Standalone points", vertices=vertices)

    assert isinstance(points.uid, UUID)
    assert points.name == "Standalone points"
    assert points.n_vertices == 3
    with pytest.raises(AttributeError, match="workspace"):
        _ = points.workspace

    np.testing.assert_allclose(points.vertices_array, vertices)
    np.testing.assert_allclose(points.locations, vertices)
    np.testing.assert_allclose(
        points.extent,
        np.array([[0.0, 0.0, 0.0], [4.0, 5.0, 6.0]]),
    )

    pickle_path = tmp_path / "points.pkl"
    with pickle_path.open("wb") as file:
        pickle.dump(points, file)

    with pickle_path.open("rb") as file:
        recovered = pickle.load(file)

    assert isinstance(recovered, PointsModel)
    assert recovered.uid == points.uid
    assert recovered.name == points.name
    assert recovered.type_uid == points.type_uid
    np.testing.assert_allclose(recovered.vertices_array, vertices)


def test_vertices_assignment_is_validated():
    """Assignment reruns PointsModel's vertex validation."""
    points = PointsModel(vertices=[[0.0, 0.0, 0.0]])
    replacement = np.array(
        [
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
        ]
    )

    points.vertices = replacement

    assert points.n_vertices == 2
    np.testing.assert_allclose(points.vertices_array, replacement)

    with pytest.raises(ValidationError, match="shape"):
        points.vertices = np.r_[1.0, 2.0, 3.0]

    np.testing.assert_allclose(points.vertices_array, replacement)


@pytest.mark.parametrize(
    ("vertices", "message"),
    [
        (np.r_[1.0, 2.0, 3.0], "shape"),
        (np.array([["x", "y", "z"]]), "must be numeric"),
        (object(), "must be a numpy array"),
    ],
)
def test_points_model_rejects_invalid_vertices(vertices, message):
    with pytest.raises(ValidationError, match=message):
        PointsModel(vertices=vertices)


def test_points_model_default_and_structured_vertices():
    """Defaults and geoh5 structured arrays use the same validation chain."""
    with pytest.warns(UserWarning, match="No 'vertices' provided"):
        default_points = PointsModel()

    np.testing.assert_allclose(default_points.vertices_array, [[0.0, 0.0, 0.0]])

    structured = np.array(
        [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)],
        dtype=VERTICES_DTYPE,
    )
    points = PointsModel(vertices=structured)

    assert points.vertices_array.dtype == np.dtype("<f8")
    np.testing.assert_allclose(
        points.vertices_array,
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    )
    assert points.as_geoh5_vertices().dtype == VERTICES_DTYPE


def test_lazy_vertices_load_once_and_cache():
    """The first array access fetches vertices and later access uses the cache."""
    uid = uuid4()
    calls = []
    expected = np.array(
        [
            [100.0, 200.0, 300.0],
            [101.0, 201.0, 301.0],
        ]
    )

    def fetcher(entity_uid, key):
        calls.append((entity_uid, key))
        return expected

    points = PointsModel.from_array_source(
        CallableArraySource(fetcher),
        uid,
        name="Lazy points",
    )

    assert repr(points.vertices).endswith("state=lazy)")
    assert not points.vertices.is_loaded
    assert calls == []

    assert points.n_vertices == 2
    assert points.vertices.is_loaded
    assert calls == [(uid, "vertices")]

    np.testing.assert_allclose(points.vertices_array, expected)
    np.testing.assert_allclose(points.vertices[1], expected[1])
    assert points.vertices.shape == (2, 3)
    assert points.vertices.dtype == np.dtype("<f8")
    assert len(points.vertices) == 2
    assert calls == [(uid, "vertices")]


def test_lazy_array_reports_unavailable_values():
    """LazyArray distinguishes an unconfigured source from a failed fetch."""
    missing_source = LazyArray(
        source=None,
        uid=None,
        key="values",
        validator=[],
    )
    with pytest.raises(ValueError, match="no source/uid"):
        _ = missing_source.value

    uid = uuid4()
    failed_source = LazyArray(
        source=CallableArraySource(lambda _uid, _key: None),
        uid=uid,
        key="values",
        validator=[],
    )
    with pytest.raises(ValueError, match="could not be loaded"):
        _ = failed_source.value


def test_serialization_categories_load_only_datasets():
    """Scalar attributes remain eager while dataset preparation loads vertices."""
    uid = uuid4()
    calls = []
    expected = np.array(
        [
            [10.0, 20.0, 30.0],
            [11.0, 21.0, 31.0],
        ]
    )

    def fetcher(entity_uid, key):
        calls.append((entity_uid, key))
        return expected

    points = PointsModel.from_array_source(
        CallableArraySource(fetcher),
        uid,
        name="Serializable points",
        metadata={"purpose": "serialization example"},
    )

    assert points.entity_type.h5_collection == "Objects"
    assert points.entity_type.h5_type_collection == "Object types"
    assert points.dataset_map == {
        "Metadata": "metadata",
        "Vertices": "vertices",
    }

    attributes = points.attributes.model_dump(by_alias=True, exclude_none=True)
    assert attributes["ID"] == uid
    assert attributes["Name"] == "Serializable points"
    assert not points.vertices.is_loaded
    assert calls == []

    datasets = points.h5_datasets()
    h5_vertices = datasets["Vertices"]

    assert datasets["Metadata"] == {"purpose": "serialization example"}
    assert h5_vertices.dtype == VERTICES_DTYPE
    np.testing.assert_allclose(
        h5_vertices.view("<f8").reshape((-1, 3)),
        expected,
    )
    assert points.vertices.is_loaded
    assert calls == [(uid, "vertices")]

    repeated = points.h5_datasets()
    assert repeated["Vertices"].dtype == VERTICES_DTYPE
    assert calls == [(uid, "vertices")]

    complete_dump = points.model_dump_everything()
    assert complete_dump["attributes"] == attributes
    assert complete_dump["datasets"]["Vertices"].dtype == VERTICES_DTYPE
    assert complete_dump["entity_type"]["ID"] == points.type_uid
    assert complete_dump["parent_uid"] is None
    assert calls == [(uid, "vertices")]

    payload = Geoh5EntityPayload.from_model(points)
    assert payload.collection == "Objects"
    assert payload.type_collection == "Object types"
    assert payload.uid == uid
    assert payload.type_uid == points.type_uid
    assert payload.attributes == attributes
    assert set(payload.datasets) == {"Metadata", "Vertices"}
    assert calls == [(uid, "vertices")]


def test_legacy_points_adapter_preserves_entity_values():
    """Legacy Points can be compared with the workspace-free model."""
    vertices = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ]
    )

    with Workspace() as workspace:
        legacy = Points.create(
            workspace,
            name="Legacy points",
            vertices=vertices,
            metadata={"source": "legacy"},
        )
        adapted = PointsModel.from_legacy_points(legacy)

        assert adapted.uid == legacy.uid
        assert adapted.name == legacy.name
        assert adapted.type_uid == legacy.entity_type.uid
        assert adapted.parent_uid == legacy.parent.uid
        assert adapted.metadata == legacy.metadata
        assert adapted.allow_move == legacy.allow_move
        assert adapted.visible == legacy.visible
        np.testing.assert_allclose(adapted.vertices_array, legacy.vertices)
