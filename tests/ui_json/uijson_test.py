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

import json

import numpy as np
import pytest

from geoh5py import Workspace
from geoh5py.objects import Curve, Points
from geoh5py.ui_json.forms import (
    BoolForm,
    DataForm,
    IntegerForm,
    ObjectForm,
    StringForm,
)
from geoh5py.ui_json.ui_json import BaseUIJson
from geoh5py.ui_json.validations import UIJsonError


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


def generate_test_uijson(workspace: Workspace, uijson, data: dict):
    return uijson(
        title="my application",
        geoh5=str(workspace.h5file),
        run_command="python -m my_module",
        monitoring_directory="my_monitoring_directory",
        conda_environment="my_conda_environment",
        workspace_geoh5=str(workspace.h5file),
        **data,
    )


def test_multiple_validations(tmp_path):
    ws = Workspace(tmp_path / "test.geoh5")
    pts = Points.create(ws, name="test", vertices=np.random.random((10, 3)))
    other_pts = pts.copy(name="other test")
    data = pts.add_data({"my_data": {"values": np.random.randn(10)}})

    class MyUIJson(BaseUIJson):
        my_object_parameter: ObjectForm
        my_other_object_parameter: ObjectForm
        my_data_parameter: DataForm

    kwargs = {
        "my_object_parameter": {
            "label": "test",
            "mesh_type": [Curve],
            "value": pts.uid,  # Wrong mesh type
        },
        "my_other_object_parameter": {
            "label": "other test",
            "mesh_type": [Points],
            "value": other_pts.uid,
        },
        "my_data_parameter": {
            "label": "data",
            "parent": "my_other_object_parameter",  # Wrong parent
            "association": "Vertex",
            "data_type": "Float",
            "dependency": "my_other_object_parameter",
            "value": data.uid,
        },
    }

    uijson = generate_test_uijson(ws, uijson=MyUIJson, data=kwargs)

    with pytest.raises(UIJsonError) as err:
        _ = uijson.to_params()

    assert "my_data_parameter data is not a child of my_other_object_parameter" in str(
        err.value
    )
    assert (
        "Object's mesh type must be one of [<class 'geoh5py.objects.curve.Curve'>]"
        in str(err.value)
    )
    assert "Dependency my_other_object_parameter must be either optional or" in str(
        err.value
    )


def test_validate_dependency_type_validation(tmp_path):
    ws = Workspace(tmp_path / "test.geoh5")

    # BoolForm dependency is valid
    class MyUIJson(BaseUIJson):
        my_parameter: BoolForm
        my_dependent_parameter: StringForm

    kwargs = {
        "my_parameter": {
            "label": "test",
            "value": True,
        },
        "my_dependent_parameter": {
            "label": "dependency",
            "value": "test",
            "dependency": "my_parameter",
        },
    }
    uijson = generate_test_uijson(ws, uijson=MyUIJson, data=kwargs)
    params = uijson.to_params()
    assert params["my_dependent_parameter"] == "test"

    # Optional non-bool dependency is valid
    class MyUIJson(BaseUIJson):
        my_parameter: StringForm
        my_dependent_parameter: StringForm

    kwargs["my_parameter"]["value"] = "not a bool"
    kwargs["my_parameter"]["optional"] = True
    uijson = generate_test_uijson(ws, uijson=MyUIJson, data=kwargs)
    params = uijson.to_params()
    assert params["my_dependent_parameter"] == "test"

    # Non-optional non-bool dependency is invalid
    kwargs["my_parameter"].pop("optional")
    msg = "Dependency my_parameter must be either optional or of boolean type"
    with pytest.raises(UIJsonError, match=msg):
        uijson = generate_test_uijson(ws, uijson=MyUIJson, data=kwargs)
        _ = uijson.to_params()


def test_parent_child_validation(tmp_path):
    ws = Workspace(tmp_path / "test.geoh5")
    pts = Points.create(ws, name="test", vertices=np.random.random((10, 3)))
    data = pts.add_data({"my_data": {"values": np.random.randn(10)}})
    other_pts = pts.copy(name="other test")

    class MyUIJson(BaseUIJson):
        my_object_parameter: ObjectForm
        my_data_parameter: DataForm

    kwargs = {
        "my_object_parameter": {
            "label": "test",
            "mesh_type": [Points],
            "value": pts.uid,
        },
        "my_data_parameter": {
            "label": "data",
            "parent": "my_object_parameter",
            "association": "Vertex",
            "data_type": "Float",
            "value": data.uid,
        },
    }

    uijson = generate_test_uijson(ws, uijson=MyUIJson, data=kwargs)
    params = uijson.to_params()
    assert params["my_data_parameter"].uid == data.uid

    # Data is not a child of the parent object
    kwargs["my_object_parameter"]["value"] = other_pts.uid
    msg = "my_data_parameter data is not a child of my_object_parameter"
    with pytest.raises(UIJsonError, match=msg):
        uijson = generate_test_uijson(ws, uijson=MyUIJson, data=kwargs)
        _ = uijson.to_params()


def test_mesh_type_validation(tmp_path):
    ws = Workspace(tmp_path / "test.geoh5")
    pts = Points.create(ws, name="test", vertices=np.random.random((10, 3)))

    class MyUIJson(BaseUIJson):
        my_object_parameter: ObjectForm

    kwargs = {
        "my_object_parameter": {
            "label": "test",
            "mesh_type": [Points],
            "value": pts.uid,
        },
    }

    uijson = generate_test_uijson(ws, uijson=MyUIJson, data=kwargs)
    params = uijson.to_params()
    assert params["my_object_parameter"].uid == pts.uid

    # Data is not a child of the parent object
    kwargs["my_object_parameter"]["mesh_type"] = [Curve]
    msg = "Object's mesh type must be one of"
    with pytest.raises(UIJsonError, match=msg):
        uijson = generate_test_uijson(ws, uijson=MyUIJson, data=kwargs)
        _ = uijson.to_params()
