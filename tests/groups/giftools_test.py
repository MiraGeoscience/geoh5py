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

import numpy as np
import pytest

from geoh5py.groups import (
    E3DInv,
    E3DMTInv,
    H3DTDInv,
    IPInv3D,
    IPOctreeInversion,
    OCTGRVDEInversion,
)
from geoh5py.groups.giftools.base import BaseGIFtoolsGroup
from geoh5py.groups.giftools.dc_octree_inversion import DCOctreeInversion
from geoh5py.groups.giftools.giftools import GIFtoolsGroup
from geoh5py.groups.giftools.gravity_gradiometry_inversion import GGInv3D
from geoh5py.groups.giftools.gravity_inversion import GZInv3D
from geoh5py.groups.giftools.magnetic_vector_inversion import MVIInv
from geoh5py.groups.giftools.magnetics_inversion import MagInv3D
from geoh5py.objects import BlockModel, Octree
from geoh5py.workspace import Workspace


def _make_octree(workspace, parent):
    """To test GIFtools groups that operate on an octree mesh."""
    return Octree.create(
        workspace,
        parent=parent,
        origin=[0, 0, 0],
        u_count=32,
        v_count=16,
        w_count=8,
        u_cell_size=1.0,
        v_cell_size=1.0,
        w_cell_size=2.0,
        rotation=45,
    )


def _make_block_model(workspace, parent):
    """To test GIFtools groups that operate on a Block Model (rectilinear/UBC tensor mesh)"""
    return BlockModel.create(
        workspace,
        parent=parent,
        origin=[0, 0, 0],
        u_cell_delimiters=np.arange(0.0, 33.0, 1.0),
        v_cell_delimiters=np.arange(0.0, 17.0, 1.0),
        z_cell_delimiters=np.arange(-16.0, 1.0, 2.0),
        rotation=45,
    )


# Test each GIFtools group can take in an appropriate object/mesh type.
@pytest.mark.parametrize(
    "group_cls, make_mesh",
    (
        (DCOctreeInversion, _make_octree),
        (IPOctreeInversion, _make_octree),
        (OCTGRVDEInversion, _make_octree),
        (E3DInv, _make_octree),
        (E3DMTInv, _make_octree),
        (MagInv3D, _make_block_model),
        (GGInv3D, _make_block_model),
        (GZInv3D, _make_block_model),
        (MVIInv, _make_block_model),
        (IPInv3D, _make_block_model),
        (H3DTDInv, _make_block_model),
    ),
)
def test_create_group(tmp_path, group_cls: type[BaseGIFtoolsGroup], make_mesh):
    h5file_path = tmp_path / r"testGroup.geoh5"
    group_name = group_cls._default_name

    # Create a workspace
    with Workspace.create(h5file_path) as workspace:
        gif = GIFtoolsGroup.create(workspace)
        group = group_cls.create(workspace, parent=gif)
        mesh = make_mesh(workspace, gif)
        group.set_parameters(mesh=mesh)
        mesh_uid = mesh.uid

    # Read the group back in
    with Workspace(h5file_path) as workspace:
        rec_obj = workspace.get_entity(group_name)[0]

        assert rec_obj.name == group_name
        assert rec_obj.parent.gif_parameters[0]["mesh"]["value"] == mesh_uid


def test_default_parameters_are_deep_copied():
    """Changing one group's nested parameters should not mutate shared defaults."""
    workspace = Workspace()

    # Create a group and change some of its parameters
    group_a = MagInv3D.create(workspace)
    group_a.set_parameters(mesh="mesh-a", cell_weight="weights-a")

    # Create another group and check that its parameters are still the default values
    group_b = MagInv3D.create(workspace, name="maginv3d_60 copy")

    assert group_b.parameters["mesh"]["value"] == ""
    assert group_b.parameters["cell_weight"]["value"] == ""

    assert MagInv3D._default_parameters["mesh"]["value"] == ""
    assert MagInv3D._default_parameters["cell_weight"]["value"] == ""
