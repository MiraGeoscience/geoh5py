# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2025-2026 Mira Geoscience Ltd.                                '
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

from pathlib import Path

import numpy as np

from geoh5py.groups import ContainerGroup
from geoh5py.objects import Points
from geoh5py.shared.utils import compare_entities
from geoh5py.ui_json.utils import monitored_directory_copy
from geoh5py.workspace import Workspace


def test_monitored_directory_copy(tmp_path: Path):
    xyz = np.random.randn(12, 3)
    values = np.random.randn(12)
    h5file_path = tmp_path / r"testPoints.geoh5"
    with Workspace.create(h5file_path) as workspace:
        group = ContainerGroup.create(workspace, name="groupee")

        points = Points.create(
            workspace, name="test", parent=group, vertices=xyz, allow_move=False
        )
        points.add_data(
            {
                "DataValues": {
                    "association": "VERTEX",
                    "values": values,
                }
            },
            property_group="property_group_test",
        )

        new_file = monitored_directory_copy(tmp_path, points)
        new_workspace = Workspace(new_file)

        assert new_workspace.get_entity("groupee")[0] is None, (
            "Parental group should not have been copied."
        )

        for entity in workspace.objects:
            # Read the data back in from a fresh workspace
            rec_entity = new_workspace.get_entity(entity.uid)[0]
            rec_data = new_workspace.get_entity(entity.children[0].uid)[0]

            compare_entities(entity, rec_entity, ignore=["_parent", "_property_table"])
            compare_entities(
                entity.children[0], rec_data, ignore=["_parent", "_property_table"]
            )

        new_file = monitored_directory_copy(tmp_path, points, copy_children=False)
        new_workspace = Workspace(new_file)

        assert len(new_workspace.data) == 0, "Child data should not have been copied."
