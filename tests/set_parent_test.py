# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2020-2026 Mira Geoscience Ltd.                                     '
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

import numpy as np

from geoh5py.groups import ContainerGroup
from geoh5py.objects import Points
from geoh5py.workspace import Workspace


def test_set_parent(tmp_path):
    # Generate a random cloud of points
    xyz = np.random.randn(2, 3)
    name = "test_points"
    h5file_path = tmp_path / r"testProject.geoh5"

    with Workspace.create(h5file_path) as workspace:
        group_a = ContainerGroup.create(workspace)
        entity = Points.create(workspace, vertices=xyz, name=name, parent=group_a)
        entity.add_data({"random": {"values": np.random.randn(xyz.shape[0])}})
        group_b = ContainerGroup.create(workspace, name="group_b")
        entity.parent = group_b

        workspace = Workspace(h5file_path)
        group_reload = workspace.get_entity("group_b")[0]
        entity_reload = workspace.get_entity(name)[0]
        data_reload = workspace.get_entity("random")[0]

        assert entity_reload.parent == group_reload, "Parent different than expected."
        assert entity_reload in group_reload.children, (
            "Entity not in the list of children."
        )
        assert data_reload in entity_reload.children, (
            "Data not in list of entity children."
        )
