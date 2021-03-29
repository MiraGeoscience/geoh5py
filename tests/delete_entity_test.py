#  Copyright (c) 2021 Mira Geoscience Ltd.
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

import tempfile
from pathlib import Path

import numpy as np

from geoh5py.groups import ContainerGroup
from geoh5py.objects import Curve
from geoh5py.workspace import Workspace


def test_delete_entities():

    xyz = np.random.randn(12, 3)
    values = np.random.randn(12)

    with tempfile.TemporaryDirectory() as tempdir:
        # Create a workspace
        workspace = Workspace(str(Path(tempdir) / r"testPoints.geoh5"))

        group = ContainerGroup.create(workspace)

        points = Curve.create(workspace, vertices=xyz, parent=group)

        data = points.add_data(
            {"DataValues": {"association": "VERTEX", "values": values}}
        )

        # Add data
        d_group = []
        for i in range(4):
            values = np.random.randn(points.n_vertices)
            d_group += [
                points.add_data(
                    {f"Period{i + 1}": {"values": values}}, property_group="myGroup"
                )
            ]

        # Property group object should have been created
        prop_group = points.get_property_group("myGroup")

        workspace.finalize()
        workspace.remove_entity(d_group[2])
        assert (
            d_group[2].uid not in prop_group.properties
        ), "Data uid was not removed from the property_group"

        workspace.remove_entity(points)
        assert (
            (len(group.children) == 0)
            and (points not in list(workspace.objects))
            and (data not in list(workspace.data))
        ), "Object and data were not fully removed from the workspace"

        workspace.remove_entity(group)
        assert group not in list(
            workspace.groups
        ), "Group object not fully remove from the workspace"
