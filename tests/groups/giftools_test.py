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

from geoh5py.groups.giftools import DCOctreeInversion, GIFtoolInversion, GIFtoolsGroup
from geoh5py.objects import Octree
from geoh5py.workspace import Workspace


@pytest.mark.parametrize("group", (DCOctreeInversion,))
def test_create_group(tmp_path, group: type[GIFtoolInversion]):
    h5file_path = tmp_path / r"testGroup.geoh5"
    group_name = group._default_name

    # Create a workspace
    with Workspace.create(h5file_path) as workspace:
        gif = GIFtoolsGroup.create(workspace)
        group = DCOctreeInversion.create(workspace, parent=gif)
        # octree = Octree.create(
        #     workspace,
        #     parent=gif,
        #     origin=[0, 0, 0],
        #     u_count=32,
        #     v_count=16,
        #     w_count=8,
        #     u_cell_size=1.0,
        #     v_cell_size=1.0,
        #     w_cell_size=2.0,
        #     rotation=45,
        # )
        # group.modify_parameters(mesh=octree)
    # Read the group back in
    rec_obj = workspace.get_entity(group_name)[0]

    assert rec_obj.name == group_name


def test_file(tmp_path):

    with Workspace(r"C:\Users\dominiquef\Documents\tests\GEOPY-807.geoh5") as ws:
        with Workspace.create(tmp_path / "test.geoh5") as new:
            gg = ws.get_entity("dcoctree_inv")[0]

            gg.parent.copy(parent=new)
            # octree = Octree.create(
            #     ws,
            #     parent=gg.parent,
            #     origin=[0, 0, 0],
            #     u_count=32,
            #     v_count=16,
            #     w_count=8,
            #     u_cell_size=1.0,
            #     v_cell_size=1.0,
            #     w_cell_size=2.0,
            #     rotation=45,
            # )
