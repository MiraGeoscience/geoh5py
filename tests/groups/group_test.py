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

import pytest

from geoh5py.groups import ContainerGroup, SimPEGGroup
from geoh5py.shared.utils import compare_entities
from geoh5py.ui_json import UIJson
from geoh5py.ui_json.forms import FloatForm
from geoh5py.workspace import Workspace


def test_create_group(tmp_path):
    h5file_path = tmp_path / r"testGroup.geoh5"
    group_name = "MyTestContainer"

    # Create a workspace
    workspace = Workspace.create(h5file_path)
    group = ContainerGroup.create(workspace, name=group_name)
    workspace.save_entity(group)

    # Read the group back in
    rec_obj = workspace.get_entity(group_name)[0]
    compare_entities(group, rec_obj)


def test_simpeg_group(tmp_path):
    h5file_path = tmp_path / r"testGroup.geoh5"

    class TestUIJson(UIJson):
        title: str = "Test UIJson"
        geoh5: Workspace | None = None
        run_command: str = "run_command"
        conda_environment: str = "mirageo"
        something: FloatForm = FloatForm(
            name="Something",
            label="Something",
            value=1.0,
        )

    # Create a workspace with group
    workspace = Workspace.create(h5file_path)
    group = SimPEGGroup.create(workspace)
    group.options = TestUIJson().serialize(mode="json", exclude_unset=False)

    # Copy
    new_workspace = Workspace.create(tmp_path / r"testGroup2.geoh5")
    group.copy(parent=new_workspace)

    # Read back in and compare
    new_workspace = Workspace(tmp_path / r"testGroup2.geoh5")
    rec_obj = new_workspace.get_entity(group.uid)[0]
    compare_entities(
        group,
        rec_obj,
        ignore=[
            "_parent",
        ],
    )


def test_add_children_group(tmp_path):
    h5file_path = tmp_path / r"testGroup.geoh5"
    group_name = "MyTestContainer"

    # Create a workspace
    workspace = Workspace.create(h5file_path)
    group = ContainerGroup.create(workspace, name=group_name)

    with pytest.raises(TypeError, match="Child must be an instance"):
        group.add_children("bidon")
