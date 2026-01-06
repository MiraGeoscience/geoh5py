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
import pytest
from h5py import File

from geoh5py.groups import ContainerGroup
from geoh5py.objects import Points
from geoh5py.shared.utils import as_str_if_uuid, compare_entities
from geoh5py.workspace import Workspace


#  pylint: disable=too-many-locals
def test_remove_root(tmp_path: Path):
    # Generate a random cloud of points
    n_data = 12
    xyz = np.random.randn(n_data, 3)
    h5file_path = tmp_path / r"testProject.geoh5"

    with Workspace.create(h5file_path) as workspace:
        group = ContainerGroup.create(workspace)
        points = Points.create(workspace, vertices=xyz, parent=group)
        data = points.add_data(
            {
                "DataValues": {
                    "association": "VERTEX",
                    "values": np.random.randn(n_data),
                },
                "DataValues2": {
                    "association": "VERTEX",
                    "values": np.random.randn(n_data),
                },
            }
        )

        group_name = "SomeGroup"

        data_group = points.add_data_to_group(data, group_name)

        # Check no crash loading existing entity
        assert workspace.load_entity(points.uid, "object")
        assert len(workspace.fetch_children(None)) == 0

    # Remove the root
    with File(h5file_path, "r+") as project:
        base = list(project.keys())[0]
        del project[base]["Root"]
        del project[base]["Groups"][as_str_if_uuid(workspace.root.uid)]
        del project[base]["Types"]["Group types"]

    # Read the data back in from a fresh workspace
    with Workspace(h5file_path) as new_workspace:
        assert len(new_workspace.fetch_children(new_workspace.root)) == 1
        rec_points = new_workspace.get_entity(points.name)[0]

        points.workspace.open()
        compare_entities(
            points,
            rec_points,
            ignore=["_parent", "_on_file", "_property_groups"],
        )
        compare_entities(
            data[0],
            new_workspace.get_entity(data[0].name)[0],
            ignore=["_parent", "_on_file"],
        )
        compare_entities(
            data_group,
            rec_points.fetch_property_group(name=group_name),
            ignore=["_parent", "_on_file"],
        )

    points.workspace.close()

    with Workspace(h5file_path) as new_workspace:
        assert len(new_workspace.list_entities_name) == 6, "Issue re-building the Root."

        points2 = Points.create(new_workspace, vertices=xyz, parent=group)
        data2 = points2.add_data(
            {
                "DataValuesb": {
                    "association": "VERTEX",
                    "values": np.random.randn(n_data),
                },
                "DataValues2b": {"values": "bidon", "association": "OBJECT"},
            }
        )

        with pytest.raises(ValueError, match="Data 'DataValues2b' association"):
            points2.add_data_to_group(data2, "bidon")

        with pytest.raises(ValueError, match="Data 'abs' not found in parent"):
            points2.add_data_to_group(["abs"], "bidon")

        points2._property_groups = None

        assert points2.remove_data_from_groups(data2) is None
