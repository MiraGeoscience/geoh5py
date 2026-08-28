# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2026 Mira Geoscience Ltd.                                     '
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

import json
from pathlib import Path
from uuid import uuid4

import h5py
import numpy as np
import pytest
from pydantic import ValidationError

from geoh5py.groups import ContainerGroup
from geoh5py.objects import Points
from geoh5py.workspace import Workspace
from geoh5py_pydantic import (
    VERTICES_DTYPE,
    Attributes,
    CallableArraySource,
    Geoh5EntityPayload,
    Geoh5Writer,
    ObjectType,
    PointsModel,
    create_geoh5,
)


def _initialize_geoh5_file(path: Path, *, with_parent: bool = False):
    """Create the project/root structure through geoh5py_pydantic."""
    create_geoh5(path)

    if with_parent:
        with Workspace(path) as workspace:
            parent = ContainerGroup.create(workspace, name="Direct writer parent")
            return parent.uid

    return None


def _as_text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else value


def test_writer_uses_legacy_variable_length_string_dtype():
    assert Geoh5Writer.string_dtype == h5py.special_dtype(vlen=str)


def test_points_model_owns_attributes_and_entity_type():
    """Nested models own HDF5 fields without losing the flat Points API."""
    uid = uuid4()
    type_uid = uuid4()
    model = PointsModel.model_validate(
        {
            "ID": uid,
            "Name": "Nested points",
            "Allow move": False,
            "Last focus": "Camera 1",
            "vertices": [[1.0, 2.0, 3.0]],
        }
    )

    assert isinstance(model.attributes, Attributes)
    assert isinstance(model.entity_type, ObjectType)
    assert model.uid == model.attributes.uid == uid
    assert model.name == model.attributes.name == "Nested points"
    assert model.type_uid == model.entity_type.uid == Points.default_type_uid()
    assert model.entity_type.h5_collection == "Objects"
    assert model.entity_type.h5_type_collection == "Object types"

    h5_attributes = model.attributes.model_dump(
        by_alias=True,
        exclude_none=True,
    )
    assert h5_attributes["Allow move"] is False
    assert h5_attributes["Last focus"] == "Camera 1"

    # Direct assignment remains available and is validated by Attributes.
    model.name = "Renamed points"
    assert model.attributes.name == "Renamed points"

    payload = Geoh5EntityPayload.from_model(model)
    assert payload.attributes["Name"] == "Renamed points"
    assert payload.type_uid == model.entity_type.uid
    assert payload.type_attributes == model.entity_type.h5_attributes()

    explicitly_nested = PointsModel(
        attributes={"Name": "Explicit attributes", "Allow move": False},
        entity_type={
            "ID": type_uid,
            "Name": "Custom points",
            "Description": "Custom type description",
        },
        vertices=[[1.0, 2.0, 3.0]],
    )
    assert explicitly_nested.name == "Explicit attributes"
    assert explicitly_nested.allow_move is False
    assert explicitly_nested.entity_type.name == "Custom points"
    assert explicitly_nested.entity_type.description == "Custom type description"


@pytest.mark.parametrize(
    ("extra_input", "expected_location"),
    [
        ({"unknown_entity_field": 1}, ("unknown_entity_field",)),
        (
            {"attributes": {"Unknown attribute": 1}},
            ("attributes", "Unknown attribute"),
        ),
        (
            {"entity_type": {"Unknown type field": 1}},
            ("entity_type", "Unknown type field"),
        ),
    ],
)
def test_points_model_rejects_unserialized_extra_fields(
    extra_input,
    expected_location,
):
    """Unknown values must fail validation rather than disappear on write."""
    with pytest.raises(ValidationError) as error:
        PointsModel.model_validate(
            {
                "vertices": [[1.0, 2.0, 3.0]],
                **extra_input,
            }
        )

    assert expected_location in {
        tuple(validation_error["loc"])
        for validation_error in error.value.errors()
        if validation_error["type"] == "extra_forbidden"
    }


def test_write_points_model(tmp_path):
    path = tmp_path / "direct_points.geoh5"
    _initialize_geoh5_file(path)

    vertices = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    model = PointsModel(
        name="Direct points",
        vertices=vertices,
        metadata={"source": "PointsModel"},
        allow_move=False,
    )

    with h5py.File(path, "r+") as h5file:
        writer = Geoh5Writer(h5file)
        object_group = writer.write(model, compression=3)
        project = writer.project
        uid = writer.format_uuid(model.uid)
        type_uid = writer.format_uuid(model.type_uid)

        # The canonical entity, shared type, and Root child entry mirror the
        # hard-link layout created by legacy write_entity/write_to_parent.
        assert object_group.id == project["Objects"][uid].id
        assert object_group["Type"].id == project["Types"]["Object types"][type_uid].id
        assert object_group.id == project["Root"]["Objects"][uid].id

        assert object_group.attrs["ID"] == uid
        assert object_group.attrs["Name"] == model.name
        assert object_group.attrs["Allow move"] == np.int8(0)
        assert object_group["Vertices"].dtype == VERTICES_DTYPE
        np.testing.assert_allclose(
            object_group["Vertices"][:].view("<f8").reshape((-1, 3)), vertices
        )

        metadata = json.loads(_as_text(object_group["Metadata"][0]))
        assert metadata == model.metadata
        assert object_group["Vertices"].compression == "gzip"
        assert object_group["Vertices"].compression_opts == 3

        with pytest.raises(FileExistsError, match="already exists"):
            writer.write(model)

    # use existing geoh5py to check things were written properly.
    with Workspace(path) as workspace:
        recovered = workspace.get_entity(model.uid)[0]
        assert isinstance(recovered, Points)
        assert recovered.name == model.name
        assert recovered.entity_type.uid == model.type_uid
        assert recovered.metadata == model.metadata
        np.testing.assert_allclose(recovered.vertices, vertices)


def test_write_lazy_points_to_explicit_parent(tmp_path):
    path = tmp_path / "lazy_points.geoh5"
    parent_uid = _initialize_geoh5_file(path, with_parent=True)
    uid = uuid4()
    calls = []
    expected_vertices = np.array([[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]])

    def fetcher(entity_uid, key):
        # simulates IO accessing the array
        calls.append((entity_uid, key))
        return expected_vertices

    model = PointsModel.from_array_source(
        CallableArraySource(fetcher),
        uid,
        name="Lazy direct points",
        parent_uid=parent_uid,
    )
    assert not model.vertices.is_loaded

    with h5py.File(path, "r+") as h5file:
        writer = Geoh5Writer(h5file)
        object_group = writer.write(model)
        uid_text = writer.format_uuid(model.uid)
        parent_text = writer.format_uuid(parent_uid)

        assert (
            object_group.id
            == writer.project["Groups"][parent_text]["Objects"][uid_text].id
        )

    # Payload creation serializes vertices once, the LazyArray then owns the cache.
    assert model.vertices.is_loaded
    assert calls == [(uid, "vertices")]


def test_write_generic_group_payload(tmp_path):
    """The format writer can consume a payload without knowing a model class."""
    path = tmp_path / "generic_group.geoh5"
    _initialize_geoh5_file(path)
    uid = uuid4()
    type_uid = uuid4()
    payload = Geoh5EntityPayload(
        collection="Groups",
        type_collection="Group types",
        uid=uid,
        type_uid=type_uid,
        parent_uid=None,
        attributes={
            "ID": uid,
            "Name": "Payload group",
            "Allow delete": True,
        },
        datasets={"Metadata": {"created_by": "generic payload"}},
        type_attributes={
            "Description": "Payload group type",
            "ID": type_uid,
            "Name": "Payload group type",
        },
    )

    with h5py.File(path, "r+") as h5file:
        writer = Geoh5Writer(h5file)
        group = writer.write_payload(payload)
        uid_text = writer.format_uuid(uid)
        type_uid_text = writer.format_uuid(type_uid)

        assert {"Data", "Groups", "Objects", "Metadata", "Type"} == set(group)
        assert group.id == writer.project["Root"]["Groups"][uid_text].id
        assert (
            group["Type"].id == writer.project["Types"]["Group types"][type_uid_text].id
        )


def test_failed_dataset_write_does_not_create_entity_type(tmp_path):
    """Entity encoding failures must not leave a new shared type behind."""
    path = tmp_path / "failed_group.geoh5"
    _initialize_geoh5_file(path)
    uid = uuid4()
    type_uid = uuid4()
    payload = Geoh5EntityPayload(
        collection="Groups",
        type_collection="Group types",
        uid=uid,
        type_uid=type_uid,
        parent_uid=None,
        attributes={"ID": uid, "Name": "Invalid payload group"},
        datasets={"Unsupported": 1},
        type_attributes={
            "Description": "Unused group type",
            "ID": type_uid,
            "Name": "Unused group type",
        },
    )

    with h5py.File(path, "r+") as h5file:
        writer = Geoh5Writer(h5file)
        uid_text = writer.format_uuid(uid)
        type_uid_text = writer.format_uuid(type_uid)

        with pytest.raises(TypeError, match="unsupported value type"):
            writer.write_payload(payload)

        assert uid_text not in writer.project["Groups"]
        assert type_uid_text not in writer.project["Types"]["Group types"]


def test_failed_type_write_removes_entity_and_new_type(tmp_path):
    """Type encoding failures must roll back both newly created groups."""
    path = tmp_path / "failed_type.geoh5"
    _initialize_geoh5_file(path)
    uid = uuid4()
    type_uid = uuid4()
    payload = Geoh5EntityPayload(
        collection="Groups",
        type_collection="Group types",
        uid=uid,
        type_uid=type_uid,
        parent_uid=None,
        attributes={"ID": uid, "Name": "Invalid type payload group"},
        datasets={},
        type_attributes={
            "ID": type_uid,
            "Name": "Invalid group type",
            "Unsupported": object(),
        },
    )

    with h5py.File(path, "r+") as h5file:
        writer = Geoh5Writer(h5file)
        uid_text = writer.format_uuid(uid)
        type_uid_text = writer.format_uuid(type_uid)

        with pytest.raises(TypeError, match="unsupported value type"):
            writer.write_payload(payload)

        assert uid_text not in writer.project["Groups"]
        assert type_uid_text not in writer.project["Types"]["Group types"]


def test_failed_parent_link_removes_entity_and_new_type(tmp_path):
    """Failures after type creation must still remove the new shared type."""
    path = tmp_path / "failed_parent_link.geoh5"
    _initialize_geoh5_file(path)
    uid = uuid4()
    type_uid = uuid4()
    payload = Geoh5EntityPayload(
        collection="Groups",
        type_collection="Group types",
        uid=uid,
        type_uid=type_uid,
        parent_uid=None,
        attributes={"ID": uid, "Name": "Conflicting child"},
        datasets={},
        type_attributes={
            "ID": type_uid,
            "Name": "Temporary group type",
        },
    )

    with h5py.File(path, "r+") as h5file:
        writer = Geoh5Writer(h5file)
        uid_text = writer.format_uuid(uid)
        type_uid_text = writer.format_uuid(type_uid)
        root_groups = writer.project["Root"].require_group("Groups")
        root_groups.create_group(uid_text)

        with pytest.raises(OSError, match="name already exists"):
            writer.write_payload(payload)

        assert uid_text not in writer.project["Groups"]
        assert type_uid_text not in writer.project["Types"]["Group types"]
        assert uid_text in root_groups
