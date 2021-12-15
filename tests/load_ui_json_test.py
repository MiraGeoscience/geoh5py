#  Copyright (c) 2021 Mira Geoscience Ltd.
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
import uuid
from copy import deepcopy
from os import path

import numpy as np
import pytest

import geoh5py.ui_json.templates as tmp
from geoh5py.groups import ContainerGroup
from geoh5py.objects import Points
from geoh5py.ui_json.constants import default_ui_json
from geoh5py.ui_json.exceptions import (
    JSONParameterValidationError,
    RequiredValidationError,
    TypeValidationError,
    UUIDStringValidationError,
    UUIDValidationError,
)
from geoh5py.ui_json.input_file import InputFile
from geoh5py.workspace import Workspace


def test_load_ui_json(tmp_path):
    xyz = np.random.randn(12, 3)
    workspace = Workspace(path.join(tmp_path, "testPoints.geoh5"))
    group = ContainerGroup.create(workspace)
    points = Points.create(workspace, vertices=xyz, parent=group)
    data = points.add_data(
        {
            "values A": {"values": np.random.randn(12)},
            "values B": {"values": np.random.randn(12)},
        }
    )
    points.add_data_to_group(data, name="My group")

    ui_json = {}
    in_file = InputFile(ui_json=ui_json)

    with pytest.raises(RequiredValidationError) as error:
        data = in_file.data

    assert "Missing 'title'" in str(error)

    ui_json = deepcopy(default_ui_json)
    ui_json["geoh5"] = 123

    in_file = InputFile(ui_json=ui_json)
    with pytest.raises(TypeValidationError) as error:
        data = in_file.data

    assert "Type 'int' provided for 'geoh5' is invalid" in str(error)

    ui_json["geoh5"] = workspace
    ui_json["object"] = tmp.object_parameter()
    del ui_json["object"]["value"]
    with pytest.raises(JSONParameterValidationError) as error:
        InputFile(ui_json=ui_json)

    assert (
        "Malformed ui.json dictionary for parameter 'object'. Missing 'value'"
        in str(error)
    )

    ui_json["object"]["value"] = str(points.uid)
    # with pytest.raises(JSONParameterValidationError) as error:
    in_file = InputFile(ui_json=ui_json)

    assert in_file.data["object"] == points.uid, "Promotion of uuid from string failed"

    # Test for value fail
    ui_json["data"] = tmp.data_parameter()
    ui_json["data"]["parent"] = points.uid
    ui_json["data"]["value"] = "goat"
    in_file = InputFile(ui_json=ui_json, validations={"data": {"uuid": []}})

    with pytest.raises(UUIDStringValidationError) as error:
        data = in_file.data

    assert "Parameter 'data' with value 'goat' is not a valid uuid string" in str(error)

    ui_json["data"]["value"] = uuid.uuid4()
    in_file = InputFile(ui_json=ui_json, validations={"data": {"uuid": [uuid.uuid4()]}})

    with pytest.raises(UUIDValidationError) as error:
        data = in_file.data

    assert "provided for 'data' is invalid.  Must be" in str(error)

    # input_data["data_group"] = tmp.data_parameter()
    # input_data["logical"] = tmp.bool_parameter()
    # input_data["choices"] = tmp.choice_string_parameter()
    # input_data["file"] = tmp.file_parameter()
    # input_data["float"] = tmp.float_parameter()
    # input_data["integer"] = tmp.integer_parameter()
