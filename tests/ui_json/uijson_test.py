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

import numpy as np

from geoh5py import Workspace
from geoh5py.objects import Points
from geoh5py.ui_json import InputFile
from geoh5py.ui_json.parameters import (
    BoolParameter,
    ChoiceStringParameter,
    DataParameter,
    DataValueParameter,
    FileParameter,
    FloatParameter,
    IntegerParameter,
    ObjectParameter,
    Parameter,
    StringParameter,
)
from geoh5py.ui_json.ui_json import UIJson


def generate_sample_uijson(testpath):
    """Returns a defaulted UIJson with all parameter types and valid data."""
    workspace = Workspace(testpath / "test.geoh5")
    pts = np.random.random((10, 3))
    data_object = Points.create(workspace, name="survey", vertices=pts)
    data = data_object.add_data(
        {
            "Bx": {"values": np.random.random(10)},
            "By": {"values": np.random.random(10)},
            "elevation": {"values": np.random.random(10)},
        }
    )

    standard_uijson_parameters = [
        Parameter("title", value="my application"),
        Parameter("geoh5"),
        Parameter("run_command"),
        BoolParameter(
            "run_command_boolean",
            label="Run python module",
            value=False,
            tooltip="Warning: launches process to run python model on save",
            main=True,
        ),
        Parameter("monitoring_directory"),
        Parameter("conda_environment"),
        Parameter("conda_environment_boolean", value=False),
        Parameter("workspace"),
    ]
    custom_uijson_parameters = [
        StringParameter("name", label="Name", value="test"),
        BoolParameter(
            "flip_sign",
            label="Flip sign",
            value=False,
        ),
        IntegerParameter("number_of_iterations", label="Number of iterations", value=5),
        FloatParameter("tolerance", label="Tolerance", value=1e-5),
        ChoiceStringParameter(
            "method", label="Method", choice_list=["cg", "ssor", "jacobi"], value="cg"
        ),
        ObjectParameter(
            "data_object",
            label="Survey",
            mesh_type=["202c5db1-a56d-4004-9cad-baafd8899406"],
            value=None,
        ),
        DataValueParameter(
            "elevation",
            label="Elevation",
            parent="data_object",
            association="Vertex",
            data_type="Float",
            is_value=True,
            property=None,
            value=1000.0,
        ),
        DataParameter(
            "x_channel",
            label="Bx",
            parent="data_object",
            association="Vertex",
            data_type="Float",
            value=None,
        ),
        DataParameter(
            "y_channel",
            label="By",
            parent="data_object",
            association="Vertex",
            data_type="Float",
            value=None,
        ),
        FileParameter("data_path", label="Data path", value=None),
    ]
    parameters = standard_uijson_parameters + custom_uijson_parameters
    uijson = UIJson(parameters)
    template = uijson.to_dict()
    ifile = InputFile(ui_json=template, validate=False)
    # ifile.write_ui_json("test.ui.json", testpath)

    return uijson


def test_uijson_construct_default_and_update(tmp_path):
    uijson = generate_sample_uijson(tmp_path)
    forms = uijson.to_dict()
    assert isinstance(forms, dict)
