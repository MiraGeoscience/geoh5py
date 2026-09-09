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

from uuid import uuid4

import numpy as np
import pytest

from geoh5py.groups import FusionTableGroup
from geoh5py.objects import Points
from geoh5py.ui_json import constants, templates
from geoh5py.workspace import Workspace


def create_data(workspace, n_data=100):
    """Create a points object with some data."""
    points = Points.create(
        workspace, name="MyPoints", vertices=np.random.rand(n_data, 3)
    )

    for i in range(5):
        points.add_data({f"MyData_{i}": {"values": np.random.randn(n_data)}})

    value_map = {0: "Unknown", 1: "positive", 2: "negative", 3: "test"}

    ref_data = points.add_data(
        {
            "DataValues": {
                "type": "referenced",
                "values": np.random.randint(0, high=3, size=n_data),
                "value_map": value_map,
            }
        }
    )
    return points, ref_data


def test_create_group(tmp_path):
    h5file_path = tmp_path / f"{__name__}.geoh5"
    group_name = "MyTestContainer"

    # Create a workspace
    with Workspace.create(h5file_path) as workspace:
        obj, reference = create_data(workspace)

        with pytest.raises(TypeError, match="must be a list or None"):
            FusionTableGroup.create(
                workspace,
                name=group_name,
                mesh=obj,
                reference_data=reference,
                feature_list="abc",
            )

        group = FusionTableGroup.create(
            workspace, name=group_name, mesh=obj, reference_data=reference
        )
        obj.parent = group

        with pytest.raises(TypeError, match="must be a UUID or Entity"):
            group.mesh = "abc"

        with pytest.raises(ValueError, match="must be a valid entity in the workspace"):
            group.mesh = uuid4()

        with pytest.raises(TypeError, match="must be an integer or None"):
            group.negative_index = "abc"

        with pytest.raises(TypeError, match="must be an integer or None"):
            group.positive_index = "abc"

        with pytest.raises(TypeError, match="must be an integer or None"):
            group.test_index = "abc"

        with pytest.raises(TypeError, match="must be a UUID or None"):
            group.reference_data = "abc"

        with pytest.raises(
            ValueError, match="be a valid ReferenceData entity in the mesh"
        ):
            group.reference_data = uuid4()

        # Testing set_active
        target = obj.children[0]
        group.set_active(False, target)
        assert any(
            elem["id"] == target.uid and elem["isChecked"] is False
            for elem in group.feature_list
        )
        with pytest.raises(TypeError, match="must be a NumericData object"):
            group.set_active(True, "abc")

    # Read the group back in
    with Workspace(h5file_path) as workspace:
        rec_obj = workspace.get_entity(group_name)[0]

        assert rec_obj.mesh == obj.uid
        assert rec_obj.reference_data == reference.uid
