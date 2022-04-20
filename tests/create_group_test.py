#  Copyright (c) 2022 Mira Geoscience Ltd.
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

import os

from geoh5py.groups import ContainerGroup, SimPEGGroup
from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace


def test_create_group(tmp_path):

    h5file_path = os.path.join(tmp_path, "testGroup.geoh5")
    group_name = "MyTestContainer"

    # Create a workspace
    workspace = Workspace(h5file_path)
    group = ContainerGroup.create(workspace, name=group_name)
    workspace.save_entity(group)
    workspace.finalize()

    # Read the group back in
    rec_obj = workspace.get_entity(group_name)[0]
    compare_entities(group, rec_obj)


def test_simpeg_group(tmp_path):

    h5file_path = os.path.join(tmp_path, "testGroup.geoh5")

    # Create a workspace
    workspace = Workspace(h5file_path)
    group = SimPEGGroup.create(workspace)
    group.options = {"run_command": "abc"}
    workspace.finalize()

    # Read the group back in
    rec_obj = workspace.get_entity(group.uid)[0]
    compare_entities(group, rec_obj)
