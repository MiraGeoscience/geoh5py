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

from pathlib import Path

import numpy as np

from geoh5py.groups import ContainerGroup
from geoh5py.objects import Points
from geoh5py.workspace import Workspace


def test_create_point_data(tmp_path: Path):
    h5file_path = tmp_path / f"{__name__}.geoh5"
    with Workspace.create(h5file_path) as workspace:
        group = ContainerGroup.create(workspace, parent=None)
        assert group.parent == workspace.root, (
            "Assigned parent=None should default to Root."
        )

        group = ContainerGroup.create(workspace)
        assert group.parent == workspace.root, (
            "Creation without parent should default to Root."
        )

        points = Points.create(workspace, parent=group)

        assert points.parent == group, "Parent setter did not work."


def test_parent_extent(tmp_path: Path):
    h5file_path = tmp_path / f"{__name__}.geoh5"

    with Workspace.create(h5file_path) as workspace:
        group = ContainerGroup.create(workspace, parent=None)

        assert group.extent is None, "Extent should be None for empty group."

        point_a = Points.create(
            workspace, vertices=np.random.randn(12, 3), parent=group
        )

        assert np.allclose(group.extent, point_a.extent), (
            "Group extent should match child extent."
        )

        point_b = Points.create(
            workspace, vertices=np.random.randn(12, 3), parent=group
        )

        lim_min = np.vstack((point_a.extent, point_b.extent)).min(axis=0)
        lim_max = np.vstack((point_a.extent, point_b.extent)).max(axis=0)
        assert np.allclose(group.extent, np.vstack([lim_min, lim_max])), (
            "Group extent should match child extent."
        )


def test_name_increment(tmp_path: Path):
    h5file_path = tmp_path / f"{__name__}.geoh5"

    with Workspace.create(h5file_path) as workspace:
        ContainerGroup.create(workspace, name="myGroup")

        # Name increment if group in same parent
        group = ContainerGroup.create(workspace, name="myGroup")
        assert group.name == "myGroup(1)"

        # No increment if group in different parent
        other_group = ContainerGroup.create(workspace, name="myGroup", parent=group)
        assert other_group.name == "myGroup"

        point_a = Points.create(
            workspace, vertices=np.random.randn(12, 3), parent=group, name="myPoint"
        )

        # Name increment if point in same parent
        point_b = Points.create(
            workspace, vertices=np.random.randn(12, 3), parent=group, name="myPoint"
        )
        assert point_b.name == "myPoint(1)"

        # No increment if point in different parent
        other_point = Points.create(
            workspace,
            vertices=np.random.randn(12, 3),
            parent=other_group,
            name="myPoint",
        )
        assert other_point.name == "myPoint"

        # Name increment if data in same parent
        point_a.add_data(
            {
                "myData": {"values": np.random.randn(12)},
            }
        )

        data_a = point_a.add_data(
            {
                "myData": {"values": np.random.randn(12)},
            }
        )

        assert data_a.name == "myData(1)"

        # No increment if data in different parent
        data_b = other_point.add_data(
            {
                "myData": {"values": np.random.randn(12)},
            }
        )
        assert data_b.name == "myData"
