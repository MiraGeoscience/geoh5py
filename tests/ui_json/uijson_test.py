#  Copyright (c) 2023 Mira Geoscience Ltd.
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

import json
from pathlib import Path

import numpy as np
import pytest

from geoh5py import Workspace
from geoh5py.objects import Points
from geoh5py.shared.exceptions import RequiredUIJsonParameterValidationError
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
    Parameter,
    RestrictedParameter,
    StringFormParameter,
)
from geoh5py.ui_json.parameters import BoolParameter, StringParameter
from geoh5py.ui_json.ui_json import UIJson


def generate_sample_uijson_data(testpath):
    workspace = Workspace(testpath / "test.geoh5")
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

    standard_uijson_parameters = [
        RestrictedParameter("title", "my application", value="my application"),
        StringParameter("geoh5"),
        StringParameter("run_command"),
        BoolFormParameter(
            "run_command_boolean",
            label="Run python module",
            value=False,
            tooltip="Warning: launches process to run python model on save",
            main=True,
        ),
        StringParameter("monitoring_directory"),
        StringParameter("conda_environment"),
        BoolParameter("conda_environment_boolean"),
        StringParameter("workspace"),
    ]
    custom_uijson_parameters = [
        StringFormParameter("name", main=True, label="Name", value="test"),
        BoolFormParameter(
            "flip_sign",
            main=True,
            label="Flip sign",
            value=False,
        ),
        IntegerFormParameter(
            "number_of_iterations", main=True, label="Number of iterations", value=5
        ),
        FloatFormParameter("tolerance", main=True, label="Tolerance", value=1e-5),
        ChoiceStringFormParameter(
            "method",
            main=True,
            label="Method",
            choice_list=["cg", "ssor", "jacobi"],
            value="cg",
        ),
        ObjectFormParameter(
            "data_object",
            main=True,
            label="Survey",
            mesh_type=["202c5db1-a56d-4004-9cad-baafd8899406"],
            value=None,
        ),
        DataValueFormParameter(
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
        DataFormParameter(
            "x_channel",
            main=True,
            label="Bx",
            parent="data_object",
            association="Vertex",
            data_type="Float",
            value=None,
        ),
        DataFormParameter(
            "y_channel",
            main=True,
            label="By",
            parent="data_object",
            association="Vertex",
            data_type="Float",
            optional=True,
            enabled=False,
            value=None,
        ),
        FileFormParameter("data_path", main=True, label="Data path", value=None),
    ]
    parameters = standard_uijson_parameters + custom_uijson_parameters
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
    workspace = Workspace(tmp_path / "test.geoh5")
    uijson = UIJson(
        [
            StringParameter("title", value="my application"),
            Parameter("geoh5", value=workspace),
        ]
    )
    assert uijson.name == "my application"
    uijson.parameters = uijson.parameters[1:]
    assert uijson.name == "test"
    uijson.parameters = []
    assert uijson.name == "uijson"


def test_uijson_validations():
    uijson = generate_sample_defaulted_uijson()
    uijson.parameters = uijson.parameters[1:]
    msg = r"UIJson: 'my application' is missing required parameter\(s\): \['title'\]."
    with pytest.raises(RequiredUIJsonParameterValidationError, match=msg):
        uijson.validate()


def test_uijson_construct_default_and_update(tmp_path):
    uijson = generate_sample_defaulted_uijson()
    filename = write_uijson(tmp_path, uijson)
    workspace, data_object = generate_sample_uijson_data(tmp_path)
    parameter_updates = {
        "name": "my test name",
        "flip_sign": True,
        "number_of_iterations": 20,
        "tolerance": 1e-6,
        "method": "ssor",
        "elevation": {
            "isValue": False,
            "property": str(data_object.get_data("elevation")[0].uid),
        },
        "x_channel": str(data_object.get_data("Bx")[0].uid),
        "y_channel": {"value": str(data_object.get_data("By")[0].uid), "enabled": True},
        "data_path": "my_data_path",
    }

    populate_sample_uijson(filename, workspace, data_object, parameter_updates)
