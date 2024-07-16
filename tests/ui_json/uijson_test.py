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
import uuid
from pathlib import Path

import numpy as np
import pytest

from geoh5py import Workspace
from geoh5py.objects import Points
from geoh5py.shared.exceptions import (
    RequiredObjectDataValidationError,
    RequiredUIJsonParameterValidationError,
)
from geoh5py.ui_json import InputFile
from geoh5py.ui_json.forms import (
    BoolFormParameter,
    ChoiceStringFormParameter,
    DataFormParameter,
    DataValueFormParameter,
    FileFormParameter,
    FloatFormParameter,
    IntegerFormParameter,
    ObjectFormParameter,
    StringFormParameter,
)
from geoh5py.ui_json.parameters import (
    BoolParameter,
    StringParameter,
    ValueRestrictedParameter,
    WorkspaceParameter,
)
from geoh5py.ui_json.ui_json import UIJson


def generate_sample_uijson_data(testpath):
    with Workspace.create(testpath / "test.geoh5") as workspace:
        pts = np.random.random((10, 3))
        data_object = Points.create(workspace, name="survey", vertices=pts)
        _ = data_object.add_data(
            {
                "Bx": {"values": np.random.random(10)},
                "By": {"values": np.random.random(10)},
                "elevation": {"values": np.random.random(10)},
            }
        )
    return workspace, data_object


def generate_sample_defaulted_uijson():
    """Returns a defaulted UIJson with all parameter types and valid data."""

    standard_uijson_parameters = {
        "title": ValueRestrictedParameter(
            "title", "my application", value="my application"
        ),
        "geoh5": WorkspaceParameter("geoh5"),
        "run_command": StringParameter("run_command"),
        "run_command_boolean": BoolFormParameter(
            "run_command_boolean",
            label="Run python module",
            value=False,
            tooltip="Warning: launches process to run python model on save",
            main=True,
        ),
        "monitoring_directory": StringParameter("monitoring_directory"),
        "conda_environment": StringParameter("conda_environment"),
        "conda_environment_boolean": BoolParameter("conda_environment_boolean"),
        "workspace": WorkspaceParameter("workspace"),
    }
    custom_uijson_parameters = {
        "save_name": StringFormParameter(
            "save_name", main=True, label="Save as", value="test"
        ),
        "flip_sign": BoolFormParameter(
            "flip_sign",
            main=True,
            label="Flip sign",
            value=False,
        ),
        "number_of_iterations": IntegerFormParameter(
            "number_of_iterations", main=True, label="Number of iterations", value=5
        ),
        "tolerance": FloatFormParameter(
            "tolerance", main=True, label="Tolerance", value=1e-5
        ),
        "method": ChoiceStringFormParameter(
            "method",
            main=True,
            label="Method",
            choice_list=["cg", "ssor", "jacobi"],
            value="cg",
        ),
        "data_object": ObjectFormParameter(
            "data_object",
            main=True,
            label="Survey",
            mesh_type=["202c5db1-a56d-4004-9cad-baafd8899406"],
            value=None,
        ),
        "elevation": DataValueFormParameter(
            "elevation",
            main=True,
            label="Elevation",
            parent="data_object",
            association="Vertex",
            data_type="Float",
            is_value=True,
            property=None,
            value=1000.0,
        ),
        "x_channel": DataFormParameter(
            "x_channel",
            main=True,
            label="Bx",
            parent="data_object",
            association="Vertex",
            data_type="Float",
            value=None,
        ),
        "y_channel": DataFormParameter(
            "y_channel",
            main=True,
            label="channel",
            group="By",
            parent="data_object",
            association="Vertex",
            data_type="Float",
            optional=True,
            enabled=False,
            value=None,
        ),
        "y_channel_uncertainty": FloatFormParameter(
            "y_channel_uncertainty",
            main=True,
            label="Uncertainty",
            group="By",
            dependency="y_channel",
            dependency_type="enabled",
            value=4.0,
        ),
        "data_path": FileFormParameter(
            "data_path", main=True, label="Data path", value=None
        ),
    }
    parameters = dict(standard_uijson_parameters, **custom_uijson_parameters)
    uijson = UIJson(parameters)

    return uijson


def write_uijson(testpath, uijson):
    template = uijson.to_dict(naming="camel")
    ifile = InputFile(ui_json=template, validate=False)
    ifile.write_ui_json("test.ui.json", testpath)

    return ifile.path_name


def populate_sample_uijson(
    default_uijson_file, workspace, data_object, parameter_updates=None
):
    with open(default_uijson_file, encoding="utf8") as file:
        data = json.load(file)
        data["geoh5"] = str(workspace.h5file)
        data["data_object"]["value"] = str(data_object.uid)

        if parameter_updates is not None:
            for key, value in parameter_updates.items():
                if isinstance(value, dict):
                    data[key].update(value)
                elif isinstance(data[key], dict):
                    data[key]["value"] = value
                else:
                    data[key] = value

    populated_file = Path(default_uijson_file).parent / "populated.ui.json"
    with open(populated_file, "w", encoding="utf8") as file:
        json.dump(data, file, indent=4)

    return populated_file


def test_uijson_name(tmp_path):
    uijson = generate_sample_defaulted_uijson()
    assert uijson.name == "my application"
    workspace = Workspace(tmp_path / "test.geoh5")
    uijson.geoh5 = workspace
    name = uijson.name
    assert name == "test"


def test_uijson_value_access():
    uijson = generate_sample_defaulted_uijson()
    assert "title" in uijson.parameters
    assert uijson.title == "my application"  # pylint: disable=no-member
    assert uijson.elevation == 1000.0  # pylint: disable=no-member
    uijson.parameters["elevation"].is_value = False
    assert uijson.elevation is None  # pylint: disable=no-member


def test_uijson_validations(tmp_path):
    uijson = generate_sample_defaulted_uijson()
    uijson.parameters = {k: v for k, v in uijson.parameters.items() if k != "title"}
    uijson.geoh5 = Workspace.create(tmp_path / "test.geoh5")
    msg = r"UIJson: 'my application' is missing required parameter\(s\): \['title'\]."
    with pytest.raises(RequiredUIJsonParameterValidationError, match=msg):
        uijson.validate()


def test_getattr():
    uijson = generate_sample_defaulted_uijson()
    assert uijson.title == "my application"
    assert uijson.parameters["title"].value == "my application"
    assert uijson.enforcers


def test_uijson_update_raises_error():
    uijson = generate_sample_defaulted_uijson()
    uijson.parameters["title"] = "some title"
    with pytest.raises(ValueError):
        uijson.update({"title": {"value": "new title"}})


def test_uijson_extra_data():
    uijson = generate_sample_defaulted_uijson()
    uijson.update({"extra_data": "some data"})
    assert uijson.parameters["extra_data"] == "some data"


def test_uijson_construct_default_and_update(tmp_path):
    uijson = generate_sample_defaulted_uijson()
    filename = write_uijson(tmp_path, uijson)
    workspace, data_object = generate_sample_uijson_data(tmp_path)
    parameter_updates = {
        "save_name": "my test name",
        "flip_sign": True,
        "number_of_iterations": 20,
        "tolerance": {"value": 1e-6, "units": "m"},
        "method": {"value": "ssor", "tooltip": "Solver types."},
        "elevation": {
            "isValue": False,
            "property": str(data_object.get_data("elevation")[0].uid),
        },
        "x_channel": str(data_object.get_data("Bx")[0].uid),
        "y_channel": {"value": str(data_object.get_data("By")[0].uid), "enabled": True},
        "data_path": "my_data_path",
    }

    populated_file = populate_sample_uijson(
        filename, workspace, data_object, parameter_updates
    )
    ifile = InputFile.read_ui_json(populated_file)

    uijson.update(ifile.ui_json)
    uijson.validate()
    forms = uijson.to_dict(naming="camel")
    assert forms["save_name"]["value"] == "my test name"
    assert forms["flip_sign"]["value"]
    assert forms["number_of_iterations"]["value"] == 20
    assert forms["tolerance"]["value"] == 1e-6
    assert forms["tolerance"]["units"] == "m"
    assert forms["method"]["value"] == "ssor"
    assert not forms["elevation"]["isValue"]
    assert (
        forms["elevation"]["property"].uid == data_object.get_data("elevation")[0].uid
    )
    assert forms["x_channel"]["value"].uid == data_object.get_data("Bx")[0].uid
    assert forms["y_channel"]["value"].uid == data_object.get_data("By")[0].uid
    assert forms["y_channel"]["enabled"]
    assert forms["data_path"]["value"] == "my_data_path"


def test_uijson_raises_required_object_data_validation_error(tmp_path):
    uijson = generate_sample_defaulted_uijson()
    filename = write_uijson(tmp_path, uijson)
    workspace, data_object = generate_sample_uijson_data(tmp_path)

    with workspace.open():
        pts = Points.create(
            workspace, vertices=np.random.rand(10, 3), name="other object"
        )
        wrong_data = pts.add_data({"wrong data": {"values": np.random.rand(10)}})

    parameter_updates = {"x_channel": str(wrong_data.uid)}
    populated_file = populate_sample_uijson(
        filename, workspace, data_object, parameter_updates
    )

    ifile = InputFile.read_ui_json(
        populated_file,
        validate=False,
    )
    with ifile.geoh5.open() as workspace:
        for param, value in ifile.ui_json.items():
            if isinstance(value, uuid.UUID):
                ifile.data[param] = workspace.get_entity(value)

    uijson.update(ifile.ui_json)
    msg = (
        r"Workspace: 'test' object\(s\) \['data_object'\] are missing "
        r"required children \['x_channel'\]."
    )
    with pytest.raises(RequiredObjectDataValidationError, match=msg):
        uijson.validate()


def test_validations():
    uijson = generate_sample_defaulted_uijson()
    assert (
        k in uijson.enforcers.validations
        for k in ["required", "required_uijson_parameters"]
    )
