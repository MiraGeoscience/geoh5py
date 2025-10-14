# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2025 Mira Geoscience Ltd.                                     '
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

import uuid

import numpy as np
import pytest

from geoh5py.objects import Curve
from geoh5py.workspace import Workspace


def test_attribute_validations():
    workspace = Workspace()
    xyz = np.random.randn(10, 3)

    with pytest.raises(TypeError, match="Input uid must be a string or uuid"):
        Curve.create(workspace, vertices=xyz, uid="abc")

    with pytest.raises(TypeError, match="Attribute 'on_file' must be of type bool"):
        Curve.create(workspace, vertices=xyz, on_file="abc")

    with pytest.raises(
        TypeError, match="Input clipping_ids must be a list of uuid.UUID or None"
    ):
        Curve.create(workspace, vertices=xyz, clipping_ids=["abc", uuid.uuid4()])

    with pytest.raises(
        TypeError, match="Input clipping_ids must be a list of uuid.UUID or None"
    ):
        Curve.create(workspace, vertices=xyz, clipping_ids="abc")


@pytest.mark.parametrize("parameter", ["partially_hidden", "public", "visible"])
def test_attribute_change(parameter):
    workspace = Workspace()
    xyz = np.random.randn(10, 3)
    curve = Curve.create(workspace, vertices=xyz)

    default = getattr(curve, parameter)
    setattr(curve, parameter, not default)

    assert getattr(curve, parameter) != default


def test_single_sibling_visibility(tmp_path):
    with Workspace.create(tmp_path / f"{__name__}.geoh5") as workspace:
        xyz = np.random.randn(10, 3)
        curve = Curve.create(workspace, vertices=xyz)

        data = curve.add_data(
            {
                "Period1": {"values": np.random.rand(10)},
                "Period2": {"values": np.random.rand(10)},
            }
        )

        data[0].visible = True
        data[1].visible = True

        assert data[1].visible is True
        assert data[0].visible is False

    with Workspace(tmp_path / f"{__name__}.geoh5", mode="r") as workspace:
        data = workspace.get_entity("Period2")[0]
        assert data.visible is True
