#  Copyright (c) 2022 Mira Geoscience Ltd.
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

from geoh5py.groups import ContainerGroup, PropertyGroup
from geoh5py.objects import Points
from geoh5py.shared import Entity
from geoh5py.shared.exceptions import (
    AssociationValidationError,
    JSONParameterValidationError,
    PropertyGroupValidationError,
    RequiredValidationError,
    ShapeValidationError,
    TypeValidationError,
    UUIDStringValidationError,
    UUIDValidationError,
)
from geoh5py.shared.utils import compare_entities
from geoh5py.ui_json import templates
from geoh5py.ui_json.constants import default_ui_json
from geoh5py.ui_json.input_file import InputFile
from geoh5py.workspace import Workspace


def get_workspace(directory):
    workspace = Workspace(path.join(directory, "..", "testPoints.geoh5"))
    if len(workspace.objects) == 0:
        xyz = np.random.randn(12, 3)
        group = ContainerGroup.create(workspace)
        points = Points.create(workspace, vertices=xyz, parent=group, name="Points_A")
        data = points.add_data(
            {
                "values A": {"values": np.random.randn(12)},
                "values B": {"values": np.random.randn(12)},
            }
        )
        points.add_data_to_group(data, name="My group")

        points_b = points.copy(copy_children=True)
        points_b.name = "Points_B"
        points_b.add_data_to_group(points_b.children, name="My group2")

        workspace.finalize()

    return workspace


def test_input_file_json():
    # Test missing required ui_json parameter
    with pytest.raises(ValueError) as error:
        InputFile(ui_json=123)

    assert "Input 'ui_json' must be of type dict or None" in str(error)

    ui_json = {}
    in_file = InputFile(ui_json=ui_json)

    with pytest.raises(RequiredValidationError) as error:
        getattr(in_file, "data")

    assert "Missing 'title'" in str(error)

    # Test wrong type for core geoh5 parameter
    ui_json = deepcopy(default_ui_json)
    ui_json["geoh5"] = 123

    in_file = InputFile(ui_json=ui_json)
    with pytest.raises(ValueError) as error:
        getattr(in_file, "data")

    assert (
        "Input 'workspace' must be a valid :obj:`geoh5py.workspace.Workspace`"
        in str(error)
    )


def test_bool_parameter():
    ui_json = deepcopy(default_ui_json)
    ui_json["logic"] = templates.bool_parameter()
    ui_json["logic"]["value"] = True
    in_file = InputFile(ui_json=ui_json)

    with pytest.raises(TypeValidationError) as error:
        in_file.validators.validate("logic", 1234)

    assert "Type 'int' provided for 'logic' is invalid.  Must be: 'bool'" in str(error)


def test_uuid_string_parameter():
    ui_json = deepcopy(default_ui_json)
    in_file = InputFile(ui_json=ui_json)

    with pytest.raises(UUIDStringValidationError) as error:
        in_file.uuid_validator("object", "hello world")

    assert (
        "Parameter 'object' with value 'hello world' is not a valid uuid string."
        in str(error)
    )


def test_shape_parameter():
    ui_json = deepcopy(default_ui_json)
    ui_json["data"] = templates.string_parameter()
    ui_json["data"]["value"] = "2,5,6,7"
    in_file = InputFile(ui_json=ui_json, validations={"data": {"shape": 3}})

    with pytest.raises(ShapeValidationError) as error:
        getattr(in_file, "data")

    assert (
        "Parameter 'data': 'value' with shape '4' was provided. Expected len(3,)."
        in str(error)
    )


def test_object_data_selection(tmp_path):
    workspace = get_workspace(tmp_path)
    points = workspace.get_entity("Points_A")[0]
    points_b = workspace.get_entity("Points_B")[0]

    ui_json = deepcopy(default_ui_json)
    ui_json["object"] = templates.object_parameter()
    ui_json["geoh5"] = workspace

    # Test missing required field from ui_json

    del ui_json["object"]["value"]
    with pytest.raises(JSONParameterValidationError) as error:
        InputFile(ui_json=ui_json)

    assert (
        "Malformed ui.json dictionary for parameter 'object'. Missing 'value'"
        in str(error)
    )

    # Test promotion on ui_json setter
    ui_json["object"]["value"] = str(points.uid)
    ui_json["object"]["meshType"] = [points.entity_type.uid]

    in_file = InputFile(ui_json=ui_json)

    assert (
        in_file.data["object"] == points
    ), "Promotion of entity from uuid string failed."

    with pytest.raises(ValueError) as error:
        in_file.data = 123

    assert "Input 'data' must be of type dict or None." in str(error)

    # Test for invalid uuid string
    ui_json["data"] = templates.data_parameter()
    ui_json["data"]["parent"] = "object"
    ui_json["data"]["value"] = "Hello World"
    in_file = InputFile(ui_json=ui_json)

    with pytest.raises(TypeValidationError) as error:
        getattr(in_file, "data")

    assert (
        "Type 'str' provided for 'data' is invalid.  Must be one of: 'UUID', 'Entity'."
        in str(error)
    )

    # Test valid uuid in workspace
    ui_json["data"]["value"] = uuid.uuid4()
    in_file = InputFile(ui_json=ui_json)

    with pytest.raises(UUIDValidationError) as error:
        getattr(in_file, "data")

    assert "provided for 'data' is invalid. Not in the list" in str(error)

    # Test data with wrong parent
    ui_json["data"]["value"] = points_b.children[0].uid
    in_file = InputFile(ui_json=ui_json)

    with pytest.raises(AssociationValidationError) as error:
        getattr(in_file, "data")

    assert "must be a child entity of parent" in str(error)

    # Test property group with wrong type
    ui_json["data"]["value"] = points.property_groups[0].uid
    ui_json["data"]["dataGroupType"] = "ABC"

    with pytest.raises(JSONParameterValidationError) as error:
        InputFile(ui_json=ui_json)

    assert (
        "Malformed ui.json dictionary for parameter 'data'. "
        "Value 'ABC' provided for 'dataGroupType' is invalid."
    ) in str(error)

    ui_json["data"]["dataGroupType"] = "3D vector"

    in_file = InputFile(ui_json=ui_json)

    with pytest.raises(PropertyGroupValidationError) as error:
        getattr(in_file, "data")

    assert (
        "Property group for 'data' must be of type '3D vector'. "
        "Provided 'My group' of type 'Multi-element'" in str(error)
    )
    ui_json["data"]["dataGroupType"] = "Multi-element"

    # Test
    in_file = InputFile()
    with pytest.raises(AttributeError) as error:
        in_file.write_ui_json(name="test", path=tmp_path)

    assert (
        "The input file requires 'ui_json' and 'data' to be set before writing out."
        in str(error)
    )

    in_file = InputFile(ui_json=ui_json)

    out_file = in_file.write_ui_json(name="test", path=tmp_path)

    with pytest.raises(ValueError) as error:
        InputFile.read_ui_json("somefile.json")

    assert "Input file should have the extension *.ui.json" in str(error)

    # Load the input back in
    reload_input = InputFile.read_ui_json(out_file)

    with pytest.raises(TypeError) as error:
        reload_input.validations = "abc"

    assert "Input validations must be of type 'dict' or None." in str(error)

    for key, value in in_file.data.items():
        if key == "geoh5":
            continue
        if isinstance(value, (Entity, PropertyGroup)):
            compare_entities(
                reload_input.data[key], value, ignore=["_parent", "_property_groups"]
            )
        elif reload_input.data[key] != value:
            raise ValueError(f"Input '{key}' differs from the output.")

    # ui_json["data"]["data_group_type"] = "Multi-element"
    # in_file = InputFile(ui_json=ui_json, validations={"data": {"property_group": points}})
    #
    # getattr(in_file, "data")

    # input_data["data_group"] = templates.data_parameter()
    # input_data["logical"] = templates.bool_parameter()
    # input_data["choices"] = templates.choice_string_parameter()
    # input_data["file"] = templates.file_parameter()
    # input_data["float"] = templates.float_parameter()
    # input_data["integer"] = templates.integer_parameter()
