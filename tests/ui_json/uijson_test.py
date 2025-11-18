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
import logging

import numpy as np
import pytest

from geoh5py import Workspace
from geoh5py.objects import Curve, Points
from geoh5py.ui_json.annotations import Deprecated
from geoh5py.ui_json.forms import (
    BoolForm,
    DataForm,
    DataOrValueForm,
    FloatForm,
    IntegerForm,
    MultiSelectDataForm,
    ObjectForm,
    RadioLabelForm,
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
                    "version": "0.1.0",
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
                        "value": str(data.uid),
                    },
                    "my_data_or_value_parameter": {
                        "label": "My other data parameter",
                        "parent": "my_object_parameter",
                        "association": "Vertex",
                        "data_type": "Float",
                        "is_value": True,
                        "property": "",
                        "value": 0.0,
                    },
                    "my_multi_select_data_parameter": {
                        "label": "My multi-select data parameter",
                        "parent": "my_other_object_parameter",
                        "association": "Vertex",
                        "data_type": "Float",
                        "value": [str(data.uid)],
                        "multi_select": True,
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
        my_data_or_value_parameter: DataOrValueForm
        my_multi_select_data_parameter: MultiSelectDataForm
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
        version="0.1.0",
        title="my application",
        geoh5=str(workspace.h5file),
        run_command="python -m my_module",
        monitoring_directory="my_monitoring_directory",
        conda_environment="my_conda_environment",
        workspace_geoh5=str(workspace.h5file),
        **data,
    )


def test_allow_extra(tmp_path):
    ws = Workspace(tmp_path / "test.geoh5")

    class MyUIJson(BaseUIJson):
        my_string_parameter: StringForm

    kwargs = {
        "my_string_parameter": {"label": "some string", "value": "test"},
        "my_extra_form_parameter": {"label": "this is extra", "value": "extra"},
        "my_extra_parameter": "this is also extra",
    }
    uijson = generate_test_uijson(ws, uijson=MyUIJson, data=kwargs)
    assert "my_extra_parameter" in uijson.model_extra
    assert "my_extra_form_parameter" in uijson.model_extra
    dump = uijson.model_dump()
    assert dump["my_extra_parameter"] == "this is also extra"
    assert dump["my_extra_form_parameter"] == {
        "label": "this is extra",
        "value": "extra",
    }


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


def test_deprecated_annotation(tmp_path, caplog):
    geoh5 = Workspace(tmp_path / "test.geoh5")

    class MyUIJson(BaseUIJson):
        my_parameter: Deprecated

    with caplog.at_level(logging.WARNING):
        _ = MyUIJson(
            version="0.1.0",
            title="my application",
            geoh5=geoh5.h5file,
            run_command="python -m my_module",
            monitoring_directory="my_monitoring_directory",
            conda_environment="my_conda_environment",
            workspace_geoh5="",
            my_parameter="whoopsie",
        )
    assert "Skipping deprecated field: my_parameter." in caplog.text


def test_grouped_forms(tmp_path):
    class MyUIJson(BaseUIJson):
        my_param: IntegerForm
        my_grouped_param: FloatForm
        my_other_grouped_param: FloatForm

    kwargs = {
        "my_param": {
            "label": "a",
            "value": 1,
        },
        "my_grouped_param": {
            "label": "b",
            "group": "my_group",
            "value": 1.0,
        },
        "my_other_grouped_param": {
            "label": "c",
            "group": "my_group",
            "value": 2.0,
        },
    }

    with Workspace(tmp_path / "test.geoh5") as ws:
        uijson = generate_test_uijson(ws, uijson=MyUIJson, data=kwargs)

    groups = uijson.groups
    assert "my_group" in groups
    assert "my_grouped_param" in groups["my_group"]
    assert "my_other_grouped_param" in groups["my_group"]


def test_disabled_forms(tmp_path):
    class MyUIJson(BaseUIJson):
        my_param: IntegerForm
        my_other_param: IntegerForm
        my_grouped_param: FloatForm
        my_other_grouped_param: FloatForm
        my_group_disabled_param: FloatForm
        my_other_group_disabled_param: FloatForm

    kwargs = {
        "my_param": {
            "label": "a",
            "value": 1,
        },
        "my_other_param": {
            "label": "b",
            "value": 2,
            "enabled": False,
        },
        "my_grouped_param": {
            "label": "c",
            "group": "my_group",
            "value": 1.0,
        },
        "my_other_grouped_param": {
            "label": "d",
            "group": "my_group",
            "value": 2.0,
        },
        "my_group_disabled_param": {
            "label": "e",
            "group": "my_other_group",
            "group_optional": True,
            "enabled": False,
            "value": 3.0,
        },
        "my_other_group_disabled_param": {
            "label": "f",
            "group": "my_other_group",
            "value": 4.0,
        },
    }

    with Workspace(tmp_path / "test.geoh5") as ws:
        uijson = generate_test_uijson(ws, uijson=MyUIJson, data=kwargs)

    assert not uijson.is_disabled("my_param")
    assert uijson.is_disabled("my_other_param")
    assert not uijson.is_disabled("my_grouped_param")
    assert not uijson.is_disabled("my_other_grouped_param")
    assert uijson.is_disabled("my_group_disabled_param")
    assert uijson.is_disabled("my_other_group_disabled_param")

    params = uijson.to_params()
    assert "my_param" in params
    assert "my_other_param" not in params


def test_unknown_uijson(tmp_path):
    ws = Workspace.create(tmp_path / "test.geoh5")
    pts = Points.create(ws, name="my points", vertices=np.random.random((10, 3)))
    data = pts.add_data({"my data": {"values": np.random.random(10)}})
    kwargs = {
        "version": "0.1.0",
        "title": "my application",
        "geoh5": str(tmp_path / "test.geoh5"),
        "run_command": "python -m my_module",
        "monitoring_directory": None,
        "conda_environment": "test",
        "workspace_geoh5": None,
        "my_string_parameter": {
            "label": "my string parameter",
            "value": "my string value",
        },
        "my_radio_label_parameter": {
            "label": "my radio label parameter",
            "original_label": "option 1",
            "alternate_label": "option 2",
            "value": "option_1",
        },
        "my_integer_parameter": {
            "label": "my integer parameter",
            "value": 10,
        },
        "my_object_parameter": {
            "label": "my object parameter",
            "mesh_type": "{202C5DB1-A56D-4004-9CAD-BAAFD8899406}",
            "value": str(pts.uid),
        },
        "my_data_parameter": {
            "label": "My data parameter",
            "parent": "my_object_parameter",
            "association": "Vertex",
            "data_type": "Float",
            "value": str(data.uid),
        },
        "my_data_or_value_parameter": {
            "label": "My other data parameter",
            "parent": "my_object_parameter",
            "association": "Vertex",
            "data_type": "Float",
            "is_value": True,
            "property": "",
            "value": 0.0,
        },
        "my_multi_choice_data_parameter": {
            "label": "My multi-choice data parameter",
            "parent": "my_object_parameter",
            "association": "Vertex",
            "data_type": "Float",
            "value": [str(data.uid)],
            "multi_select": True,
        },
        "my_optional_parameter": {
            "label": "my optional parameter",
            "value": 2.0,
            "optional": True,
            "enabled": False,
        },
        "my_group_optional_parameter": {
            "label": "my group optional parameter",
            "value": 3.0,
            "group": "my group",
            "group_optional": True,
            "enabled": False,
        },
        "my_grouped_parameter": {
            "label": "my grouped parameter",
            "value": 4.0,
            "group": "my group",
        },
    }
    with open(tmp_path / "test.ui.json", mode="w", encoding="utf8") as file:
        file.write(json.dumps(kwargs))
    uijson = BaseUIJson.read(tmp_path / "test.ui.json")

    assert isinstance(uijson.my_string_parameter, StringForm)
    assert isinstance(uijson.my_radio_label_parameter, RadioLabelForm)
    assert isinstance(uijson.my_integer_parameter, IntegerForm)
    assert isinstance(uijson.my_object_parameter, ObjectForm)
    assert isinstance(uijson.my_data_parameter, DataForm)
    assert isinstance(uijson.my_data_or_value_parameter, DataOrValueForm)
    assert isinstance(uijson.my_multi_choice_data_parameter, MultiSelectDataForm)
    params = uijson.to_params()
    assert params["my_object_parameter"].uid == pts.uid
    assert params["my_data_parameter"].uid == data.uid
    assert params["my_data_or_value_parameter"] == 0.0
    assert params["my_multi_choice_data_parameter"][0].uid == data.uid
    assert "my_optional_parameter" not in params
    assert "my_group_optional_parameter" not in params
    assert "my_grouped_parameter" not in params


def test_str_and_repr(tmp_path):
    Workspace.create(tmp_path / "test.geoh5")
    uijson = BaseUIJson(
        version="0.1.0",
        title="my application",
        geoh5=str(tmp_path / "test.geoh5"),
        run_command="python -m my_module",
        monitoring_directory=None,
        conda_environment="test",
        workspace_geoh5=None,
    )
    str_uijson = str(uijson)
    repr_uijson = repr(uijson)
    assert "UIJson('my application')" in repr_uijson
    assert '"version": "0.1.0"' in str_uijson
    uijson.write(tmp_path / "test.ui.json")
    str_uijson = str(uijson)
    repr_uijson = repr(uijson)
    assert "UIJson('test.ui.json')" in repr_uijson
    assert '"version": "0.1.0"' in str_uijson
