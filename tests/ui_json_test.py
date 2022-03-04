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

import json
from copy import deepcopy
from os import path
from uuid import uuid4

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
    ValueValidationError,
)
from geoh5py.shared.utils import compare_entities
from geoh5py.ui_json import InputValidation, templates
from geoh5py.ui_json.constants import default_ui_json, ui_validations
from geoh5py.ui_json.input_file import InputFile
from geoh5py.ui_json.utils import collect
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
    with pytest.raises(ValueError) as excinfo:
        InputFile(ui_json=123)

    assert "Input 'ui_json' must be of type dict or None" in str(excinfo)

    ui_json = {"test": 4}
    in_file = InputFile(ui_json=ui_json)

    with pytest.raises(RequiredValidationError) as excinfo:
        getattr(in_file, "data")

    assert RequiredValidationError.message("title", None, None) == str(excinfo.value)

    # Test wrong type for core geoh5 parameter
    ui_json = deepcopy(default_ui_json)
    ui_json["geoh5"] = 123

    in_file = InputFile(ui_json=ui_json)
    with pytest.raises(ValueError) as excinfo:
        getattr(in_file, "data")

    assert (
        "Input 'workspace' must be a valid :obj:`geoh5py.workspace.Workspace`"
        in str(excinfo)
    )


def test_optional_parameter():
    test = templates.optional_parameter("enabled")
    assert test["optional"]
    assert test["enabled"]
    test = templates.optional_parameter("disabled")
    assert test["optional"]
    assert not test["enabled"]


def test_bool_parameter():

    ui_json = deepcopy(default_ui_json)
    ui_json["logic"] = templates.bool_parameter()
    ui_json["logic"]["value"] = True
    in_file = InputFile(ui_json=ui_json)

    with pytest.raises(TypeValidationError) as excinfo:
        in_file.validators.validate("logic", 1234)

    assert TypeValidationError.message("logic", "int", ["bool"]) == str(excinfo.value)


def test_integer_parameter(tmp_path):

    workspace = get_workspace(tmp_path)
    ui_json = deepcopy(default_ui_json)
    ui_json["geoh5"] = workspace
    ui_json["integer"] = templates.integer_parameter()
    in_file = InputFile(ui_json=ui_json)
    data = in_file.data
    data["integer"] = 4.0

    with pytest.raises(TypeValidationError) as excinfo:
        in_file.data = data
    assert TypeValidationError.message("integer", "float", ["int"]) == str(
        excinfo.value
    )

    data["integer"] = 123
    in_file.data = data

    out_file = in_file.write_ui_json()
    reload_input = InputFile.read_ui_json(out_file)

    assert (
        reload_input.data["integer"] == 123
    ), "IntegerParameter did not properly save to file."

    test = templates.integer_parameter(optional="enabled")
    assert test["optional"]
    assert test["enabled"]


def test_float_parameter(tmp_path):

    workspace = get_workspace(tmp_path)
    ui_json = deepcopy(default_ui_json)
    ui_json["geoh5"] = workspace
    ui_json["float_parameter"] = templates.float_parameter()
    in_file = InputFile(ui_json=ui_json)
    data = in_file.data
    data["float_parameter"] = 4

    with pytest.raises(TypeValidationError) as excinfo:
        in_file.data = data
    assert TypeValidationError.message("float_parameter", "int", ["float"]) == str(
        excinfo.value
    )

    data["float_parameter"] = 123.0
    in_file.data = data

    out_file = in_file.write_ui_json()
    reload_input = InputFile.read_ui_json(out_file)

    assert (
        reload_input.data["float_parameter"] == 123.0
    ), "IntegerParameter did not properly save to file."

    test = templates.float_parameter(optional="enabled")
    assert test["optional"]
    assert test["enabled"]


def test_string_parameter(tmp_path):

    workspace = get_workspace(tmp_path)
    ui_json = deepcopy(default_ui_json)
    ui_json["geoh5"] = workspace
    ui_json["string_parameter"] = templates.string_parameter()
    in_file = InputFile(ui_json=ui_json)
    data = in_file.data
    data["string_parameter"] = 4

    with pytest.raises(TypeValidationError) as excinfo:
        in_file.data = data
    assert TypeValidationError.message("string_parameter", "int", ["str"]) == str(
        excinfo.value
    )

    data["string_parameter"] = "goodtogo"
    in_file.data = data

    out_file = in_file.write_ui_json()
    reload_input = InputFile.read_ui_json(out_file)

    assert (
        reload_input.data["string_parameter"] == "goodtogo"
    ), "IntegerParameter did not properly save to file."

    test = templates.string_parameter(optional="enabled")
    assert test["optional"]
    assert test["enabled"]


def test_choice_string_parameter(tmp_path):

    workspace = get_workspace(tmp_path)
    ui_json = deepcopy(default_ui_json)
    ui_json["geoh5"] = workspace
    ui_json["choice_string_parameter"] = templates.choice_string_parameter()
    in_file = InputFile(ui_json=ui_json)
    data = in_file.data
    data["choice_string_parameter"] = "Option C"

    with pytest.raises(ValueValidationError) as excinfo:
        in_file.data = data
    assert ValueValidationError.message(
        "choice_string_parameter", "Option C", ["Option A", "Option B"]
    ) == str(excinfo.value)

    data["choice_string_parameter"] = "Option A"
    in_file.data = data

    out_file = in_file.write_ui_json()
    reload_input = InputFile.read_ui_json(out_file)

    assert (
        reload_input.data["choice_string_parameter"] == "Option A"
    ), "IntegerParameter did not properly save to file."

    test = templates.choice_string_parameter(optional="enabled")
    assert test["optional"]
    assert test["enabled"]


def test_file_parameter():
    test = templates.file_parameter(optional="enabled")
    assert test["optional"]
    assert test["enabled"]


def test_shape_parameter(tmp_path):

    workspace = get_workspace(tmp_path)
    ui_json = deepcopy(default_ui_json)
    ui_json["data"] = templates.string_parameter(value="2,5,6,7")
    ui_json["geoh5"] = workspace
    in_file = InputFile(ui_json=ui_json, validations={"data": {"shape": (3,)}})

    with pytest.raises(ShapeValidationError) as excinfo:
        getattr(in_file, "data")

    assert ShapeValidationError.message("data", (4,), (3,)) == str(excinfo.value)


def test_missing_required_field(tmp_path):

    workspace = get_workspace(tmp_path)
    ui_json = deepcopy(default_ui_json)
    ui_json["object"] = templates.object_parameter(optional="enabled")
    assert ui_json["object"]["optional"]
    assert ui_json["object"]["enabled"]
    ui_json["geoh5"] = workspace

    del ui_json["object"]["value"]
    with pytest.raises(JSONParameterValidationError) as excinfo:
        InputFile(ui_json=ui_json)
    assert JSONParameterValidationError.message(
        "object", RequiredValidationError.message("value", None, None)
    ) == str(excinfo.value)


def test_object_promotion(tmp_path):
    workspace = get_workspace(tmp_path)
    points = workspace.get_entity("Points_A")[0]

    ui_json = deepcopy(default_ui_json)
    ui_json["object"] = templates.object_parameter()
    ui_json["geoh5"] = workspace
    ui_json["object"]["value"] = str(points.uid)
    ui_json["object"]["meshType"] = [points.entity_type.uid]

    in_file = InputFile(ui_json=ui_json)

    assert (
        in_file.data["object"] == points
    ), "Promotion of entity from uuid string failed."

    with pytest.raises(ValueError) as excinfo:
        in_file.data = 123
    assert "Input 'data' must be of type dict or None." in str(excinfo)


def test_invalid_uuid_string(tmp_path):
    workspace = get_workspace(tmp_path)
    ui_json = deepcopy(default_ui_json)
    ui_json["geoh5"] = workspace
    ui_json["data"] = templates.data_parameter(optional="enabled")
    ui_json["data"]["parent"] = "object"
    ui_json["data"]["value"] = 4
    in_file = InputFile(ui_json=ui_json)

    with pytest.raises(TypeValidationError) as excinfo:
        getattr(in_file, "data")
    assert TypeValidationError.message(
        "data", "int", ["str", "UUID", "Entity", "NoneType"]
    ) == str(excinfo.value)


def test_valid_uuid_in_workspace(tmp_path):
    workspace = get_workspace(tmp_path)
    ui_json = deepcopy(default_ui_json)
    ui_json["geoh5"] = workspace
    ui_json["data"] = templates.data_parameter()
    ui_json["data"]["parent"] = "object"
    bogus_uuid = uuid4()
    ui_json["data"]["value"] = bogus_uuid
    in_file = InputFile(ui_json=ui_json)

    with pytest.raises(AssociationValidationError) as excinfo:
        getattr(in_file, "data")
    assert AssociationValidationError.message("data", bogus_uuid, workspace) == str(
        excinfo.value
    )


def test_data_with_wrong_parent(tmp_path):
    workspace = get_workspace(tmp_path)
    points = workspace.get_entity("Points_A")[0]
    points_b = workspace.get_entity("Points_B")[0]

    ui_json = deepcopy(default_ui_json)
    ui_json["geoh5"] = workspace
    ui_json["object"] = templates.object_parameter()
    ui_json["object"]["value"] = str(points.uid)
    ui_json["object"]["meshType"] = [points.entity_type.uid]
    ui_json["data"] = templates.data_parameter()
    ui_json["data"]["parent"] = "object"
    ui_json["data"]["value"] = points_b.children[0].uid
    in_file = InputFile(ui_json=ui_json)

    with pytest.raises(AssociationValidationError) as excinfo:
        getattr(in_file, "data")
    assert AssociationValidationError.message(
        "data", points_b.children[0], points
    ) == str(excinfo.value)


def test_property_group_with_wrong_type(tmp_path):
    workspace = get_workspace(tmp_path)
    points = workspace.get_entity("Points_A")[0]

    ui_json = deepcopy(default_ui_json)
    ui_json["object"] = templates.object_parameter(optional="enabled")
    ui_json["object"]["value"] = str(points.uid)
    ui_json["geoh5"] = workspace
    ui_json["data"] = templates.data_parameter()
    ui_json["data"]["value"] = points.property_groups[0].uid
    ui_json["data"]["dataGroupType"] = "ABC"

    with pytest.raises(JSONParameterValidationError) as excinfo:
        InputFile(ui_json=ui_json)

    assert JSONParameterValidationError.message(
        "data",
        ValueValidationError.message(
            "dataGroupType", "ABC", ui_validations["dataGroupType"]["values"]
        ),
    ) == str(excinfo.value)

    ui_json["data"]["dataGroupType"] = "3D vector"
    ui_json["data"]["value"] = points.property_groups[0]
    in_file = InputFile(ui_json=ui_json)

    with pytest.raises(PropertyGroupValidationError) as excinfo:
        getattr(in_file, "data")
    assert PropertyGroupValidationError.message(
        "data", points.property_groups[0], "3D vector"
    ) == str(excinfo.value)


def test_input_file(tmp_path):
    workspace = get_workspace(tmp_path)
    ui_json = deepcopy(default_ui_json)
    ui_json["geoh5"] = workspace
    in_file = InputFile()
    with pytest.raises(AttributeError) as excinfo:
        in_file.write_ui_json(name="test", path=tmp_path)

    assert (
        "The input file requires 'ui_json' and 'data' to be set before writing out."
        in str(excinfo)
    )

    in_file = InputFile(ui_json=ui_json)
    out_file = in_file.write_ui_json(path=tmp_path)

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


def test_write_ui_json(tmp_path):
    # Make sure that none2str is applied in dict_mapper
    workspace = get_workspace(tmp_path)
    ui_json = deepcopy(default_ui_json)
    ui_json["geoh5"] = workspace
    ui_json["test"] = templates.float_parameter(optional="disabled")
    in_file = InputFile(ui_json=ui_json)
    in_file.write_ui_json(name="test_write.ui.json", path=tmp_path)
    print(in_file.data)
    with open(path.join(tmp_path, "test_write.ui.json"), encoding="utf-8") as file:
        ui_json = json.load(file)
        assert ui_json["test"]["value"] == ""


def test_data_value_parameter_a(tmp_path):
    workspace = get_workspace(tmp_path)
    ui_json = deepcopy(default_ui_json)
    ui_json["geoh5"] = workspace
    ui_json["object"] = templates.object_parameter(optional="enabled")
    ui_json["data"] = templates.data_value_parameter(
        parent="object", optional="enabled"
    )

    assert ui_json["data"]["optional"]
    assert ui_json["data"]["enabled"]

    in_file = InputFile(ui_json=ui_json)
    out_file = in_file.write_ui_json(path=tmp_path, name="ABC")
    reload_input = InputFile.read_ui_json(out_file)

    assert reload_input.data["object"] is None, "Object not reloaded as None"
    assert reload_input.data["data"] == 0.0


@pytest.mark.skip(reason="Failing on github for unknown reason")
def test_data_value_parameter_b(tmp_path):

    workspace = get_workspace(tmp_path)
    points_a = workspace.get_entity("Points_A")[0]
    data_b = workspace.get_entity("values B")[0]
    ui_json = deepcopy(default_ui_json)
    ui_json["geoh5"] = workspace
    ui_json["object"] = templates.object_parameter(value=points_a.uid)
    ui_json["data"] = templates.data_value_parameter(
        parent="object", is_value=False, prop=data_b.uid
    )

    in_file = InputFile(ui_json=ui_json)
    out_file = in_file.write_ui_json()
    reload_input = InputFile.read_ui_json(out_file)

    assert reload_input.data["object"].uid == points_a.uid
    assert reload_input.data["data"].uid == data_b.uid


def test_data_parameter(tmp_path):
    workspace = get_workspace(tmp_path)
    points_b = workspace.get_entity("Points_B")[0]

    ui_json = deepcopy(default_ui_json)
    ui_json["geoh5"] = workspace
    ui_json["object"] = templates.object_parameter(value=points_b.uid)
    ui_json["data"] = templates.data_parameter(data_group_type="Multi-element")


def test_stringify(tmp_path):
    # pylint: disable=protected-access
    workspace = get_workspace(tmp_path)
    ui_json = deepcopy(default_ui_json)
    ui_json["geoh5"] = workspace.h5file
    ui_json["test"] = templates.integer_parameter(value=None)
    in_file = InputFile(
        ui_json=ui_json, validations={"test": {"types": [int, type(None)]}}
    )

    with pytest.raises(ValueError) as excinfo:
        in_file.update_ui_values({"test": None}, none_map={"test": 4})

    assert "The following parameters are not optional." in str(
        excinfo
    ), "Failed to raise error on None with not optional."

    in_file.ui_json["test"]["optional"] = True
    in_file.ui_json["test"]["enabled"] = True

    in_file.update_ui_values({"test": None}, none_map={"test": 4})

    assert in_file.ui_json["test"]["value"] == 4
    assert not in_file.ui_json["test"]["enabled"]

    ui_json["test_group"] = templates.string_parameter(optional="enabled")
    ui_json["test_group"]["group"] = "test_group"
    ui_json["test_group"]["groupOptional"] = True
    ui_json["test"] = templates.integer_parameter(value=None)
    ui_json["test"]["group"] = "test_group"

    in_file = InputFile(ui_json=ui_json, validations={"test": {"types": [int]}})
    in_file.update_ui_values({"test": None}, none_map={"test": 4})
    assert in_file.ui_json["test"]["value"] == 4
    assert not in_file.ui_json["test"]["enabled"]
    assert not in_file.ui_json["test_group"]["enabled"]
    assert "optional" not in in_file.ui_json["test"]

    ui_json["test_group"] = templates.string_parameter(optional="enabled")
    ui_json["test_group"]["group"] = "test_group"
    ui_json["test_group"]["groupOptional"] = False
    ui_json["test"] = templates.integer_parameter(value=None)
    ui_json["test"]["group"] = "test_group"

    in_file = InputFile(
        ui_json=ui_json, validations={"test": {"types": [int, type(None)]}}
    )
    in_file.update_ui_values({"test": 2}, none_map={"test": 4})
    assert in_file.ui_json["test"]["value"] == 2
    assert in_file.ui_json["test"]["enabled"]


def test_collect():
    ui_json = deepcopy(default_ui_json)
    ui_json["string_parameter"] = templates.string_parameter(optional="enabled")
    ui_json["float_parameter"] = templates.float_parameter(optional="disabled")
    ui_json["integer_parameter"] = templates.integer_parameter(optional="enabled")
    enabled_params = collect(ui_json, "enabled", value=True)
    assert all("enabled" in v for v in enabled_params.values())
    assert all(v["enabled"] for v in enabled_params.values())


def test_unique_validations():
    # pylint: disable=protected-access
    result = InputValidation._unique_validators(
        {"param1": {"types": [str], "values": ["test2"]}, "param2": {"types": [float]}}
    )
    assert all(k in result for k in ["types", "values"])
    assert all(k in ["types", "values"] for k in result)


def test_required_validators():
    # pylint: disable=protected-access
    result = InputValidation._required_validators(
        {"param1": {"types": [str], "values": ["test2"]}, "param2": {"types": [float]}}
    )
    assert all(k in result for k in ["types", "values"])
    assert all(k in ["types", "values"] for k in result)
    assert all(k == v.validator_type for k, v in result.items())


def test_merge_validations():
    # pylint: disable=protected-access

    ui_json = deepcopy(default_ui_json)
    ui_json["string_parameter"] = templates.string_parameter(optional="enabled")
    ui_json["float_parameter"] = templates.float_parameter(optional="disabled")
    ui_json["integer_parameter"] = templates.integer_parameter()
    validations = InputValidation._validations_from_uijson(ui_json)
    validations = InputValidation._merge_validations(
        validations, {"integer_parameter": {"types": [type(None)]}}
    )
    # If validation exists it is overwritten
    assert len(validations["integer_parameter"]["types"]) == 1
    assert type(None) in validations["integer_parameter"]["types"]

    # If validation doesn't exist it is added
    validations = InputValidation._merge_validations(
        validations, {"integer_parameter": {"shape": (3, 2)}}
    )
    assert all(k in validations["integer_parameter"] for k in ["types", "shape"])
