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

from copy import deepcopy
from os import path

import numpy as np
import pytest

import geoh5py.ui_json.templates as tmp
from geoh5py.groups import ContainerGroup
from geoh5py.objects import Points
from geoh5py.ui_json.constants import default_ui_json
from geoh5py.ui_json.input_file import InputFile
from geoh5py.ui_json.validators import RequiredValidationError, TypeValidationError
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

    input_data = {"geoh5": None}

    ui_json = deepcopy(default_ui_json)
    with pytest.raises(RequiredValidationError) as error:
        InputFile(ui_json=ui_json, data=input_data)

    assert "Missing 'title'" in str(error)

    input_data["title"] = "My title"

    with pytest.raises(TypeValidationError) as error:
        InputFile(ui_json=ui_json, data=input_data)

    assert "Type 'NoneType' provided for 'geoh5' is invalid" in str(error)
    # print(input_file)

    input_data["geoh5"] = workspace
    input_data["object"] = tmp.object_parameter()
    input_data["data"] = tmp.data_value_parameter()
    input_data["data_group"] = tmp.data_parameter()
    input_data["logical"] = tmp.bool_parameter()
    input_data["choices"] = tmp.choice_string_parameter()
    input_data["file"] = tmp.file_parameter()
    input_data["float"] = tmp.float_parameter()
    input_data["integer"] = tmp.integer_parameter()
    input_data["string"] = tmp.string_parameter()

    input_data["string"]["new_input"] = []

    ui_json.update(input_data)
    with pytest.raises(TypeValidationError) as error:
        InputFile(ui_json=ui_json)

    print(error)
