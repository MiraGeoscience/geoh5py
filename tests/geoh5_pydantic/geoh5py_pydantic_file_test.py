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

import h5py
import numpy as np
import pytest
from pydantic import ValidationError

from geoh5py.workspace import Workspace
from geoh5py_pydantic import (
    ROOT_TYPE_UID,
    Geoh5Project,
    Geoh5Writer,
    RootModel,
    create_geoh5,
)


def test_project_model_separates_attributes_from_creation_options():
    """Project fields serialize either to group attributes or HDF5 options."""
    project = Geoh5Project.model_validate(
        {
            "Contributors": ["First", "Second"],
            "Distance unit": "feet",
            "GA Version": "2",
            "Version": 2.2,
            "page_size": 1024,
        }
    )

    assert project.h5_attributes() == {
        "Contributors": ("First", "Second"),
        "Distance unit": "feet",
        "GA Version": "2",
        "Version": 2.2,
    }
    assert project.h5_creation_options() == {
        "fs_strategy": "page",
        "page_buf_size": 1024 * 256,
        "fs_page_size": 1024,
        "libver": ("v110", "v114"),
    }


@pytest.mark.parametrize("page_size", [511, 513, "1024", True])
def test_project_model_rejects_invalid_page_sizes(page_size):
    with pytest.raises(ValidationError):
        Geoh5Project(page_size=page_size)


def test_root_model_owns_root_and_group_type_defaults():
    """Root defaults mirror legacy RootGroup without requiring a Workspace."""
    root = RootModel()

    assert root.name == "Workspace"
    assert root.allow_delete is False
    assert root.allow_move is False
    assert root.allow_rename is False
    assert root.parent_uid is None
    assert root.type_uid == ROOT_TYPE_UID
    assert root.entity_type.allow_move_content is True
    assert root.entity_type.allow_delete_content is True


def test_create_geoh5_writes_core_structure_and_root(tmp_path):
    """The Pydantic creation path produces the required geoh5 hard-link layout."""
    path = tmp_path / "pydantic_created.geoh5"
    project_model = Geoh5Project(contributors=("Test contributor",))
    root_model = RootModel(name="Pydantic workspace")

    assert create_geoh5(path, project=project_model, root=root_model) == path

    with h5py.File(path, "r") as h5file:
        assert list(h5file) == ["GEOSCIENCE"]
        project = h5file["GEOSCIENCE"]
        assert set(project) == {"Data", "Groups", "Objects", "Root", "Types"}
        assert set(project["Types"]) == {
            "Data types",
            "Group types",
            "Object types",
        }

        assert list(project.attrs["Contributors"]) == ["Test contributor"]
        assert project.attrs["Distance unit"] == "meter"
        assert project.attrs["GA Version"] == "1"
        assert project.attrs["Version"] == np.float64(2.1)

        root_uid = Geoh5Writer.format_uuid(root_model.uid)
        type_uid = Geoh5Writer.format_uuid(ROOT_TYPE_UID)
        canonical_root = project["Groups"][root_uid]
        root_type = project["Types"]["Group types"][type_uid]

        assert project["Root"].id == canonical_root.id
        assert canonical_root["Type"].id == root_type.id
        assert set(canonical_root) == {"Data", "Groups", "Objects", "Type"}
        assert canonical_root.attrs["Name"] == "Pydantic workspace"
        assert canonical_root.attrs["Allow delete"] == np.int8(0)
        assert root_type.attrs["Allow move contents"] == np.int8(1)
        assert root_type.attrs["Allow delete contents"] == np.int8(1)

    # Opening through the existing reader verifies semantic compatibility with
    # the current geoh5 implementation, rather than HDF5 byte-for-byte equality.
    with Workspace(path) as workspace:
        assert workspace.root.uid == root_model.uid
        assert workspace.root.name == root_model.name
        assert workspace.root.entity_type.uid == ROOT_TYPE_UID


def test_create_geoh5_does_not_replace_existing_file(tmp_path):
    path = tmp_path / "existing.geoh5"
    path.write_text("existing content", encoding="utf-8")

    with pytest.raises(FileExistsError):
        create_geoh5(path)

    assert path.read_text(encoding="utf-8") == "existing content"
