#  Copyright (c) 2024 Mira Geoscience Ltd.
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

from __future__ import annotations

import json

import numpy as np
import pytest

from geoh5py import Workspace
from geoh5py.objects import Points
from geoh5py.ui_json.forms import DataForm, IntegerForm, ObjectForm, StringForm
from geoh5py.ui_json.ui_json import BaseUIJson
from geoh5py.ui_json.validation import UIJsonError


def sample_uijson(test_path):
    uijson_path = test_path / "test.ui.json"
    geoh5_path = test_path / "test.geoh5"
    with Workspace.create(geoh5_path) as workspace:
        pts = Points.create(workspace, name="test", vertices=np.random.random((10, 3)))
        data = pts.add_data({"my data": {"values": np.random.random(10)}})
        other_pts = Points.create(
            workspace, name="other test", vertices=np.random.random((10, 3))
        )
    with open(uijson_path, mode="w", encoding="utf8") as file:
        file.write(
            json.dumps(
                {
                    "title": "my application",
                    "geoh5": str(geoh5_path),
                    "run_command": "python -m my_module",
                    "run_command_boolean": True,
                    "monitoring_directory": "my_monitoring_directory",
                    "conda_environment": "my_conda_environment",
                    "conda_environment_boolean": False,
                    "workspace_geoh5": str(geoh5_path),
                    "my_string_parameter": {
                        "label": "My string parameter",
                        "value": "my string value",
                    },
                    "my_integer_parameter": {
                        "label": "My integer parameter",
                        "value": 10,
                    },
                    "my_object_parameter": {
                        "label": "My object parameter",
                        "mesh_type": ["{202C5DB1-A56D-4004-9CAD-BAAFD8899406}"],
                        "value": str(pts.uid),
                    },
                    "my_other_object_parameter": {
                        "label": "My other object parameter",
                        "mesh_type": ["{202C5DB1-A56D-4004-9CAD-BAAFD8899406}"],
                        "value": str(other_pts.uid),
                    },
                    "my_data_parameter": {
                        "label": "My data parameter",
                        "parent": "my_object_parameter",
                        "association": "Vertex",
                        "data_type": "Float",
                        "is_value": False,
                        "property": str(data.uid),
                        "value": 0.0,
                    },
                    "my_other_data_parameter": {
                        "label": "My other data parameter",
                        "parent": "my_object_parameter",
                        "association": "Vertex",
                        "data_type": "Float",
                        "is_value": True,
                        "property": "",
                        "value": 0.0,
                    },
                    "my_faulty_data_parameter": {
                        "label": "My faulty data parameter",
                        "parent": "my_other_object_parameter",
                        "association": "Vertex",
                        "data_type": "Float",
                        "value": str(data.uid),
                    },
                    "my_absent_uid_parameter": {
                        "label": "My absent uid parameter",
                        "mesh_type": ["{202C5DB1-A56D-4004-9CAD-BAAFD8899406}"],
                        "value": "{00000000-0000-0000-0000-000000000000}",
                    },
                }
            )
        )
        return uijson_path


def test_uijson(tmp_path):
    class MyUIJson(BaseUIJson):
        my_string_parameter: StringForm
        my_integer_parameter: IntegerForm
        my_object_parameter: ObjectForm
        my_other_object_parameter: ObjectForm
        my_data_parameter: DataForm
        my_other_data_parameter: DataForm
        my_faulty_data_parameter: DataForm
        my_absent_uid_parameter: ObjectForm

    uijson = MyUIJson.read(sample_uijson(tmp_path))
    with pytest.raises(UIJsonError) as err:
        with Workspace(uijson.geoh5, mode="r+") as workspace:
            _ = uijson.to_params(workspace=workspace)

    assert "my_absent_uid_parameter" in str(err.value)
    assert "my_faulty_data_parameter" in str(err.value)
