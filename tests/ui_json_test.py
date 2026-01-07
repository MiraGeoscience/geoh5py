# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2020-2026 Mira Geoscience Ltd.                                '
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
import re
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from geoh5py.groups import ContainerGroup, DrillholeGroup, PropertyGroup
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
from geoh5py.ui_json.input_file import DEFAULT_UI_JSON_NAME, InputFile
from geoh5py.ui_json.utils import collect
from geoh5py.workspace import Workspace

from .drillhole_v4_0_test import create_drillholes


def get_workspace(directory: str | Path):
    file = Path(directory).parent / "testPoints.geoh5"

    if file.exists():
        workspace = Workspace(file)
    else:
        workspace = Workspace.create(file)

    if len(workspace.objects) == 0:
        group = ContainerGroup.create(workspace)
        DrillholeGroup.create(workspace, name="drh_group", parent=group)
        points = Points.create(
            workspace, vertices=np.random.randn(12, 3), parent=group, name="Points_A"
        )
        data = points.add_data(
            {
                "values A": {"values": np.random.randn(12)},
                "values B": {"values": np.random.randn(12)},
            }
        )
        points.add_data_to_group(data, "My group")

        points_b = points.copy(copy_children=True)
        points_b.name = "Points_B"

        no_property_child = [
            child for child in points_b.children if not isinstance(child, PropertyGroup)
        ]

        points_b.add_data_to_group(no_property_child, "My group2")

    return workspace


def test_input_file_json():
    # Test missing required ui_json parameter
    with pytest.raises(
        ValueError, match="Input 'ui_json' must be of type dict or None"
    ):
        InputFile(ui_json=123)

    with pytest.raises(
        AttributeError, match="'ui_json' must be set before setting data."
    ):
        InputFile().data = {"abc": 123}

    with pytest.raises(
        AttributeError, match="InputFile requires a 'ui_json' to be defined."
    ):
        InputFile().update_ui_values({"abc": 123})

    ui_json = {"test": 4}

    with pytest.raises(
        RequiredValidationError,
        match=RequiredValidationError.message("title", None, None),
    ):
        InputFile(ui_json=ui_json).data

    # Test wrong type for core geoh5 parameter
    ui_json = deepcopy(default_ui_json)
    ui_json["geoh5"] = 123

    with pytest.raises(
        ValueError,
        match="Input 'geoh5' must be a valid :obj:`geoh5py.workspace.Workspace`.",
    ):
        InputFile(ui_json=ui_json).data


def test_workspace_geoh5_path(tmp_path):
    workspace = get_workspace(tmp_path)
    ui_json = deepcopy(default_ui_json)
    ui_json["geoh5"] = workspace
    ui_json["workspace_geoh5"] = workspace.h5file

    in_file = InputFile(ui_json=ui_json)
    out_file = in_file.write_ui_json()
    reload_input = InputFile.read_ui_json(out_file)
    assert isinstance(reload_input.data["geoh5"], Workspace)
    assert isinstance(reload_input.data["workspace_geoh5"], str)


def test_input_file_name_path(tmp_path: Path):
    # pylint: disable=protected-access

    # Test handling of name attribute
    test = InputFile()
    test.name = "test.ui.json"
    assert test.name == "test.ui.json"  # usual behaviour

    test._name = None
    ui_json = deepcopy(default_ui_json)
    ui_json["title"] = "Jarrod"
    test._ui_json = ui_json
    assert test.name == "Jarrod.ui.json"  # ui.json extension added

    # Test handling of path attribute
    with pytest.warns(
        DeprecationWarning,
        match="The 'workspace' property is deprecated. Use 'geoh5' instead.",
    ):
        test.workspace = Workspace.create(tmp_path / r"test.geoh5")

    assert test.path == str(tmp_path)  # pulled from workspace.h5file
    test.path = tmp_path
    assert test.path == str(tmp_path)  # usual behaviour

    with pytest.raises(FileNotFoundError):
        test.path = str(tmp_path / "nonexisting")

    with pytest.raises(ValueError, match="is not a directory"):
        test.path = str(tmp_path / "test.geoh5")

    # test path_name method
    assert test.path_name == str(tmp_path / r"Jarrod.ui.json")
    test = InputFile()
    assert test.path_name is None

    with pytest.raises(AttributeError, match="requires 'path' and 'name'"):
        test.write_ui_json()


def test_optional_parameter():
    test = templates.optional_parameter("enabled")
    assert test["optional"]
    assert test["enabled"]
    test = templates.optional_parameter("disabled")
    assert test["optional"]
    assert not test["enabled"]


def test_bool_parameter(tmp_path):
    workspace = get_workspace(tmp_path)
    ui_json = deepcopy(default_ui_json)
    ui_json["geoh5"] = workspace
    ui_json["logic"] = templates.bool_parameter()
    ui_json["logic"]["value"] = True
    in_file = InputFile(ui_json=ui_json)

    with pytest.raises(
        TypeValidationError, match=TypeValidationError.message("logic", "int", ["bool"])
    ):
        in_file.validators.validate("logic", 1234)


def test_integer_parameter(tmp_path: Path):
    workspace = get_workspace(tmp_path)
    ui_json = deepcopy(default_ui_json)
    ui_json["geoh5"] = workspace
    ui_json["integer"] = templates.integer_parameter()
    in_file = InputFile(ui_json=ui_json)
    data = in_file.data
    data["integer"] = 4.0

    with pytest.raises(
        TypeValidationError,
        match=TypeValidationError.message("integer", "float", ["int"]),
    ):
        in_file.data = data

    data.pop("integer")
    with pytest.warns(UserWarning, match="The number of input values"):
        in_file.data = data

    data["integer"] = 123
    in_file.data = data
    out_file = in_file.write_ui_json()
    reload_input = InputFile.read_ui_json(out_file)

    assert reload_input.data["integer"] == 123, (
        "IntegerParameter did not properly save to file."
    )

    test = templates.integer_parameter(optional="enabled")
    assert test["optional"]
    assert test["enabled"]


def test_float_parameter(tmp_path: Path):
    workspace = get_workspace(tmp_path)
    ui_json = deepcopy(default_ui_json)
    ui_json["geoh5"] = workspace
    ui_json["float_parameter"] = templates.float_parameter()
    in_file = InputFile(ui_json=ui_json)
    data = in_file.data
    data["float_parameter"] = 4

    with pytest.raises(
        TypeValidationError,
        match=TypeValidationError.message("float_parameter", "int", ["float"]),
    ):
        in_file.data = data

    data["float_parameter"] = 123.0
    in_file.data = data

    out_file = in_file.write_ui_json()
    reload_input = InputFile.read_ui_json(out_file)

    assert reload_input.data["float_parameter"] == 123.0, (
        "IntegerParameter did not properly save to file."
    )

    test = templates.float_parameter(optional="enabled")
    assert test["optional"]
    assert test["enabled"]


def test_string_parameter(tmp_path: Path):
    workspace = get_workspace(tmp_path)
    ui_json = deepcopy(default_ui_json)
    ui_json["geoh5"] = workspace
    ui_json["string_parameter"] = templates.string_parameter()
    in_file = InputFile(ui_json=ui_json)
    data = in_file.data
    data["string_parameter"] = 4

    with pytest.raises(
        TypeValidationError,
        match=TypeValidationError.message("string_parameter", "int", ["str"]),
    ):
        in_file.data = data

    data["string_parameter"] = "goodtogo"
    in_file.data = data

    out_file = in_file.write_ui_json()
    reload_input = InputFile.read_ui_json(out_file)

    assert reload_input.data["string_parameter"] == "goodtogo", (
        "IntegerParameter did not properly save to file."
    )

    test = templates.string_parameter(optional="enabled")
    assert test["optional"]
    assert test["enabled"]


def test_choice_string_parameter(tmp_path: Path):
    workspace = get_workspace(tmp_path)
    ui_json = deepcopy(default_ui_json)
    ui_json["geoh5"] = workspace
    ui_json["choice_string_parameter"] = templates.choice_string_parameter()
    in_file = InputFile(ui_json=ui_json)
    data = in_file.data
    data["choice_string_parameter"] = "Option C"

    with pytest.raises(
        ValueValidationError,
        match=ValueValidationError.message(
            "choice_string_parameter", "Option C", ["Option A", "Option B"]
        ),
    ):
        in_file.data = data

    data["choice_string_parameter"] = "Option A"
    in_file.data = data

    out_file = in_file.write_ui_json()
    reload_input = InputFile.read_ui_json(out_file)

    assert reload_input.data["choice_string_parameter"] == "Option A", (
        "IntegerParameter did not properly save to file."
    )

    test = templates.choice_string_parameter(optional="enabled")
    assert test["optional"]
    assert test["enabled"]


def test_multi_choice_string_parameter(tmp_path):
    workspace = get_workspace(tmp_path)
    ui_json = deepcopy(default_ui_json)
    ui_json["geoh5"] = workspace
    ui_json["choice_string_parameter"] = templates.choice_string_parameter()
    ui_json["choice_string_parameter"]["multiSelect"] = True
    ui_json["choice_string_parameter"]["optional"] = True
    ui_json["choice_string_parameter"]["enabled"] = False
    ui_json["choice_string_parameter"]["value"] = ""
    in_file = InputFile(ui_json=ui_json)
    ui_json["choice_string_parameter"]["enabled"] = True
    ui_json["choice_string_parameter"]["value"] = []
    in_file = InputFile(ui_json=ui_json)
    data = in_file.data

    data["choice_string_parameter"] = ["Option C"]

    with pytest.raises(
        ValueValidationError,
        match=ValueValidationError.message(
            "choice_string_parameter", "Option C", ["Option A", "Option B"]
        ),
    ):
        in_file.data = data

    data["choice_string_parameter"] = ["Option A"]
    in_file.data = data

    out_file = in_file.write_ui_json()
    reload_input = InputFile.read_ui_json(out_file)

    assert reload_input.data["choice_string_parameter"] == ["Option A"], (
        "IntegerParameter did not properly save to file."
    )


def test_file_parameter():
    test = templates.file_parameter(optional="enabled")
    assert test["optional"]
    assert test["enabled"]


def test_shape_parameter(tmp_path: Path):
    workspace = get_workspace(tmp_path)
    ui_json = deepcopy(default_ui_json)
    ui_json["data"] = templates.string_parameter(value="2,5,6,7")
    ui_json["geoh5"] = workspace

    with pytest.raises(
        ShapeValidationError,
        match=re.escape(ShapeValidationError.message("data", (1,), (2,))),
    ):
        InputFile(ui_json=ui_json, validations={"data": {"shape": (2,)}}).data


def test_missing_required_field(tmp_path: Path):
    workspace = get_workspace(tmp_path)
    ui_json = deepcopy(default_ui_json)
    ui_json["object"] = templates.object_parameter(optional="enabled")
    assert ui_json["object"]["optional"]
    assert ui_json["object"]["enabled"]
    ui_json["geoh5"] = workspace

    del ui_json["object"]["value"]
    with pytest.raises(
        JSONParameterValidationError,
        match=JSONParameterValidationError.message(
            "object", RequiredValidationError.message("value", None, None)
        ),
    ):
        InputFile(ui_json=ui_json).data


def test_object_promotion(tmp_path: Path):
    workspace = get_workspace(tmp_path)
    points = workspace.get_entity("Points_A")[0]

    ui_json = deepcopy(default_ui_json)
    ui_json["object"] = templates.object_parameter()
    ui_json["geoh5"] = workspace
    ui_json["object"]["value"] = str(points.uid)
    ui_json["object"]["meshType"] = [points.entity_type.uid]

    in_file = InputFile(ui_json=ui_json)
    in_file.write_ui_json("test.ui.json", path=tmp_path)

    # Read back in
    new_in_file = InputFile.read_ui_json(tmp_path / "test.ui.json")
    assert new_in_file.data["object"].uid == points.uid, (
        "Promotion of entity from uuid string failed."
    )

    with pytest.raises(ValueError, match="Input 'data' must be of type dict or None."):
        new_in_file.data = 123


def test_group_promotion(tmp_path):
    workspace = get_workspace(tmp_path)
    group = workspace.get_entity("Container Group")[0]
    dh_group = workspace.get_entity("drh_group")[0]
    ui_json = deepcopy(default_ui_json)
    ui_json["object"] = templates.group_parameter()
    ui_json["geoh5"] = workspace
    ui_json["object"]["value"] = str(group.uid)
    ui_json["object"]["groupType"] = [group.entity_type.uid]

    ui_json["dh"] = templates.group_parameter(optional="enabled")
    ui_json["dh"]["value"] = str(dh_group.uid)
    ui_json["dh"]["groupType"] = [dh_group.entity_type.uid]

    in_file = InputFile(ui_json=ui_json)
    in_file.write_ui_json("test.ui.json", path=tmp_path)

    new_in_file = InputFile.read_ui_json(tmp_path / "test.ui.json")
    assert new_in_file.data["object"].uid == group.uid, (
        "Promotion of entity from uuid string failed."
    )

    assert new_in_file.data["dh"].uid == dh_group.uid, (
        "Promotion of entity from uuid string failed."
    )


def test_drillhole_group_promotion(tmp_path):
    _, workspace = create_drillholes(tmp_path)

    with workspace.open("r+"):
        dh_group = workspace.get_entity("DH_group")[0]
        ui_json = deepcopy(default_ui_json)
        ui_json["object"] = templates.group_parameter()
        ui_json["geoh5"] = workspace
        ui_json["object"]["value"] = ["interval_values_a", "text Data"]
        ui_json["object"]["groupValue"] = dh_group.uid
        ui_json["object"]["groupType"] = [dh_group.entity_type.uid]
        ui_json["object"]["multiSelect"] = True

        in_file = InputFile(ui_json=ui_json)

        assert in_file.ui_json["object"]["value"] == ["interval_values_a", "text Data"]
        data = in_file.data
        assert in_file.ui_json["object"]["groupValue"] == dh_group.uid
        assert in_file.ui_json["object"]["value"] == ["interval_values_a", "text Data"]
        assert data["object"] == {
            "group_value": dh_group,
            "value": ["interval_values_a", "text Data"],
        }

        in_file.write_ui_json("test.ui.json", path=tmp_path)

    new_in_file = InputFile.read_ui_json(tmp_path / "test.ui.json")
    assert new_in_file.data["object"]["group_value"].uid == dh_group.uid, (
        "Promotion of entity from uuid string failed."
    )

    # test_errors specific to drillholes group Values
    with workspace.open("r"):
        with pytest.raises(TypeError, match="Input value for 'group_value'"):
            in_file._update_group_value_ui("object", {"bi": "don"})


def test_invalid_uuid_string(tmp_path: Path):
    workspace = get_workspace(tmp_path)
    ui_json = deepcopy(default_ui_json)
    ui_json["geoh5"] = workspace
    ui_json["data"] = templates.data_parameter(optional="enabled")
    ui_json["data"]["parent"] = "object"
    ui_json["data"]["value"] = 4

    with pytest.raises(
        TypeValidationError,
        match=TypeValidationError.message("data", "int", ["str", "UUID", "Entity"]),
    ):
        InputFile(ui_json=ui_json).data


def test_valid_uuid_in_workspace(tmp_path: Path):
    workspace = get_workspace(tmp_path)
    ui_json = deepcopy(default_ui_json)
    ui_json["geoh5"] = workspace
    ui_json["data"] = templates.data_parameter()
    ui_json["data"]["parent"] = "object"
    bogus_uuid = uuid4()
    ui_json["data"]["value"] = bogus_uuid

    with pytest.raises(
        AssociationValidationError,
        match=AssociationValidationError.message("data", bogus_uuid, workspace),
    ):
        InputFile(ui_json=ui_json).data


def test_property_group_with_wrong_type(tmp_path: Path):
    workspace = get_workspace(tmp_path)
    points = workspace.get_entity("Points_A")[0]

    ui_json = deepcopy(default_ui_json)
    ui_json["object"] = templates.object_parameter(optional="enabled")
    ui_json["object"]["value"] = str(points.uid)
    ui_json["geoh5"] = workspace
    ui_json["data"] = templates.data_parameter()
    ui_json["data"]["value"] = points.property_groups[0].uid
    ui_json["data"]["dataGroupType"] = "ABC"

    with pytest.raises(
        JSONParameterValidationError,
        match=JSONParameterValidationError.message(
            "data",
            ValueValidationError.message(
                "dataGroupType", "ABC", ui_validations["dataGroupType"]["values"]
            ),
        ),
    ):
        InputFile(ui_json=ui_json).data

    ui_json["data"]["dataGroupType"] = "3D vector"
    ui_json["data"]["value"] = points.property_groups[0]

    with pytest.raises(
        PropertyGroupValidationError,
        match=re.escape(
            PropertyGroupValidationError.message(
                "data", points.property_groups[0], ["3D vector"]
            )
        ),
    ):
        InputFile(ui_json=ui_json).data

    ui_json["data"]["dataGroupType"] = ["3D vector", "Multi-element"]

    assert InputFile(ui_json=ui_json).data is not None


def test_data_with_wrong_parent(tmp_path: Path):
    workspace = get_workspace(tmp_path)
    points = workspace.get_entity("Points_A")[0]
    points_b = workspace.get_entity("Points_B")[0]

    no_property_child = [
        child for child in points_b.children if not isinstance(child, PropertyGroup)
    ]

    ui_json = deepcopy(default_ui_json)
    ui_json["geoh5"] = workspace
    ui_json["object"] = templates.object_parameter()
    ui_json["object"]["value"] = str(points.uid)
    ui_json["object"]["meshType"] = [points.entity_type.uid]
    ui_json["data"] = templates.data_parameter()
    ui_json["data"]["parent"] = "object"
    ui_json["data"]["value"] = no_property_child[0].uid

    with pytest.raises(
        AssociationValidationError,
        match=AssociationValidationError.message("data", no_property_child[0], points),
    ):
        InputFile(ui_json=ui_json).data


def test_input_file(tmp_path: Path):
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

    with pytest.raises(
        ValueError, match="Input file should have the extension .ui.json"
    ):
        InputFile.read_ui_json("somefile.json")

    with pytest.raises(
        TypeError,
        match="expected str, bytes or os.PathLike object|"
        + "argument should be a str or an os.PathLike object where "
        + "__fspath__ returns a str, not 'int'",
    ):
        InputFile.read_ui_json(123)

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


def test_write_ui_json(tmp_path: Path):
    # Make sure that none2str is applied in dict_mapper
    workspace = get_workspace(tmp_path)
    ui_json = deepcopy(default_ui_json)
    ui_json["geoh5"] = workspace
    ui_json["test"] = templates.float_parameter(optional="disabled")
    in_file = InputFile(ui_json=ui_json)
    in_file.write_ui_json(name="test_write.ui.json", path=tmp_path)
    with open(tmp_path / r"test_write.ui.json", encoding="utf-8") as file:
        ui_json = json.load(file)
        assert ui_json["geoh5"] == str(Path(workspace.h5file))
        assert ui_json["test"]["value"] == 1.0


def test_in_memory_geoh5(tmp_path: Path):
    workspace = Workspace()
    ui_json = deepcopy(default_ui_json)
    ui_json["geoh5"] = workspace
    ui_json["test"] = templates.float_parameter(optional="disabled")
    in_file = InputFile(ui_json=ui_json)
    in_file.write_ui_json(name="test_write.ui.json", path=tmp_path)
    with open(tmp_path / r"test_write.ui.json", encoding="utf-8") as file:
        ui_json = json.load(file)
        assert ui_json["geoh5"] == "[in-memory]"
        assert ui_json["test"]["value"] == 1.0


def test_data_value_parameter_a(tmp_path: Path):
    workspace = get_workspace(tmp_path)
    ui_json = deepcopy(default_ui_json)
    ui_json["geoh5"] = workspace
    ui_json["object"] = templates.object_parameter(optional="enabled")
    ui_json["data"] = templates.data_value_parameter(
        parent="object", optional="enabled"
    )

    assert ui_json["data"]["optional"]
    assert ui_json["data"]["enabled"]

    in_file = InputFile(ui_json=ui_json, validate=False)
    out_file = in_file.write_ui_json(path=tmp_path, name="ABC")
    reload_input = InputFile.read_ui_json(out_file)

    assert reload_input.data["object"] is None, "Object not reloaded as None"
    assert reload_input.data["data"] == 0.0


# @pytest.mark.skip(reason="Failing on github for unknown reason")
def test_data_value_parameter_b(tmp_path: Path):
    workspace = get_workspace(tmp_path)
    points_a = workspace.get_entity("Points_A")[0]
    no_property_child = [
        child for child in points_a.children if not isinstance(child, PropertyGroup)
    ]

    data_b = no_property_child[0]
    ui_json = deepcopy(default_ui_json)
    ui_json["geoh5"] = workspace
    ui_json["object"] = templates.object_parameter(value=points_a.uid)
    ui_json["data"] = templates.data_value_parameter(
        parent="object", is_value=False, prop=data_b.uid
    )

    in_file = InputFile(ui_json=ui_json)

    out_file = in_file.write_ui_json(name="test.ui.json", path=tmp_path)
    reload_input = InputFile.read_ui_json(out_file)

    assert reload_input.data["object"].uid == points_a.uid
    assert reload_input.data["data"].uid == data_b.uid

    # Change data to float and re-write
    reload_input.data["data"] = 123.0
    reload_input.write_ui_json(name="test.ui.json", path=tmp_path)
    reload_input = InputFile.read_ui_json(out_file)
    assert reload_input.data["data"] == 123.0
    assert reload_input.ui_json["data"]["isValue"]


def test_multi_object_value_parameter(tmp_path: Path):
    workspace = get_workspace(tmp_path)
    points_a = workspace.get_entity("Points_A")[0]
    points_b = workspace.get_entity("Points_B")[0]

    no_property_child = [
        child for child in points_a.children if not isinstance(child, PropertyGroup)
    ]

    data_b = no_property_child[0]
    ui_json = deepcopy(default_ui_json)
    ui_json["geoh5"] = workspace
    ui_json["object"] = templates.object_parameter(value=[], multi_select=True)
    ui_json["data"] = templates.data_value_parameter(
        parent="object", is_value=False, prop=data_b.uid
    )
    in_file = InputFile(ui_json=ui_json)

    with pytest.warns(
        UserWarning,
        match="Data associated with multiSelect dependent is not supported. Validation ignored.",
    ):
        in_file.data

    ui_json.pop("data")
    in_file = InputFile(ui_json=ui_json)
    data = in_file.data
    data["object"] = points_b.uid

    data["object"] = [points_b.uid]
    in_file.data = data

    out_file = in_file.write_ui_json()
    reload_input = InputFile.read_ui_json(out_file)
    object_b = reload_input.geoh5.get_entity("Points_B")

    assert reload_input.data["object"] == object_b, (
        "IntegerParameter did not properly save to file."
    )


def test_data_parameter(tmp_path: Path):
    workspace = get_workspace(tmp_path)
    points_b = workspace.get_entity("Points_B")[0]

    ui_json = deepcopy(default_ui_json)
    ui_json["geoh5"] = workspace
    ui_json["object"] = templates.object_parameter(value=points_b.uid)
    ui_json["data"] = templates.data_parameter(data_group_type="Multi-element")


def test_data_object_file_association(tmp_path: Path):
    workspace = get_workspace(tmp_path)
    points_b = workspace.get_entity("Points_B")[0]
    file_path = tmp_path / "test.txt"

    # create a dummy txt file
    with open(file_path, "w") as file:
        file.write("Hello World")

    file = points_b.add_file(file_path)

    ui_json = deepcopy(default_ui_json)
    ui_json["geoh5"] = workspace
    ui_json["object"] = templates.object_parameter(value=points_b.uid)
    ui_json["data"] = templates.data_parameter(
        value=file.uid, data_type="Filename", association="Object", parent="object"
    )

    in_file = InputFile(ui_json=ui_json)

    in_file.write_ui_json(name="test.ui.json", path=tmp_path)

    assert in_file.data["data"] == file


def test_stringify(tmp_path: Path):
    # pylint: disable=protected-access
    workspace = get_workspace(tmp_path)
    ui_json = deepcopy(default_ui_json)
    ui_json["geoh5"] = workspace
    ui_json["test"] = templates.integer_parameter(value=None, optional="disabled")
    in_file = InputFile(
        ui_json=ui_json, validations={"test": {"types": [int, type(None)]}}
    )

    in_file.ui_json["test"]["optional"] = True
    in_file.ui_json["test"]["enabled"] = True

    in_file.update_ui_values({"test": None})

    assert in_file.ui_json["test"]["value"] is None
    assert not in_file.ui_json["test"]["enabled"]

    ui_json["test_group"] = templates.string_parameter(optional="enabled")
    ui_json["test_group"]["group"] = "test_group"
    ui_json["test_group"]["groupOptional"] = True
    ui_json["test"] = templates.integer_parameter(value=1, optional="enabled")
    ui_json["test"]["group"] = "test_group"
    in_file = InputFile(ui_json=ui_json, validations={"test": {"types": [int]}})

    in_file.update_ui_values({"test": None})

    assert in_file.ui_json["test"]["value"] is not None
    assert not in_file.ui_json["test"]["enabled"]
    assert in_file.ui_json["test_group"]["enabled"]
    assert "optional" in in_file.ui_json["test"]

    ui_json["test_group"] = templates.string_parameter(optional="enabled")
    ui_json["test_group"]["group"] = "test_group"
    ui_json["test_group"]["groupOptional"] = False
    ui_json["test"] = templates.integer_parameter(value=None, optional="disabled")
    ui_json["test"]["group"] = "test_group"

    in_file = InputFile(
        ui_json=ui_json, validations={"test": {"types": [int, type(None)]}}
    )
    in_file.update_ui_values({"test": 2})
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
    ui_json["data_value_parameter"] = {
        "isValue": True,
        "parent": "Dwayne",
    }
    validations = InputValidation._validations_from_uijson(ui_json)
    validations = InputValidation._merge_validations(
        validations, {"integer_parameter": {"types": [type(None)]}}
    )

    # Test handling of isValue
    assert validations["data_value_parameter"]["association"] == "Dwayne"
    assert validations["data_value_parameter"]["uuid"] is None

    # If validation exists it is overwritten
    assert len(validations["integer_parameter"]["types"]) == 1
    assert type(None) in validations["integer_parameter"]["types"]

    # If validation doesn't exist it is added
    validations = InputValidation._merge_validations(
        validations, {"integer_parameter": {"shape": (3, 2)}}
    )
    assert all(k in validations["integer_parameter"] for k in ["types", "shape"])


def test_dependency_enabling(tmp_path: Path):
    workspace = get_workspace(tmp_path)
    ui_json = deepcopy(default_ui_json)
    ui_json["geoh5"] = workspace
    ui_json["parameter_a"] = templates.float_parameter(optional="enabled")

    ui_json["parameter_b"] = templates.float_parameter()
    ui_json["parameter_b"]["dependency"] = "parameter_a"
    ui_json["parameter_b"]["dependencyType"] = "enabled"
    ui_json["parameter_b"]["enabled"] = True

    in_file = InputFile(ui_json=ui_json)

    # TODO This operation should raise an error instead of a warning
    # as the parent dependency is enabled
    with pytest.warns(UserWarning, match="Non-option parameter"):
        in_file.update_ui_values({"parameter_b": None})

    # Test disabled
    ui_json["parameter_a"]["enabled"] = False
    ui_json["parameter_b"]["dependencyType"] = "disabled"
    ui_json["parameter_b"]["enabled"] = True
    ui_json["parameter_b"]["value"] = 123.0
    in_file = InputFile(ui_json=ui_json)

    in_file.write_ui_json(path=tmp_path, name="test.ui.json")

    with pytest.warns(UserWarning, match="Non-option parameter"):
        in_file.update_ui_values({"parameter_b": None})


def test_range_label(tmp_path):
    workspace = Workspace.create(tmp_path / "test.geoh5")
    points = Points.create(workspace, vertices=np.random.randn(12, 3), name="my points")
    data = points.add_data({"my data": {"values": np.random.randn(12)}})

    ui_json = deepcopy(default_ui_json)
    ui_json["geoh5"] = str(workspace.h5file)
    ui_json["object"] = templates.object_parameter(value=str(points.uid))
    ui_json["test"] = templates.range_label_template(
        value=[0.2, 0.8], parent="object", property_=data.uid, is_complement=False
    )

    ifile = InputFile(ui_json=ui_json)
    ifile.write_ui_json("test_range_label.ui.json", path=tmp_path)
    new = InputFile.read_ui_json(tmp_path / "test_range_label.ui.json")

    new_data = new.data["test"]
    new_data["property"] = new_data["property"].uid

    assert new.data["test"] == {
        "is_complement": False,
        "value": [0.2, 0.8],
        "property": data.uid,
    }


def test_default_naming(tmp_path):
    workspace = get_workspace(tmp_path)
    ui_json = deepcopy(default_ui_json)
    ui_json["geoh5"] = workspace
    ui_json["workspace_geoh5"] = workspace.h5file

    in_file = InputFile(ui_json=ui_json)
    in_file.name = DEFAULT_UI_JSON_NAME

    assert in_file.name == "Custom_UI.ui.json"


def test_copy_relatives(tmp_path):
    workspace = get_workspace(tmp_path)
    original_points = workspace.get_entity("Points_A")[0]

    ui_json = deepcopy(default_ui_json)
    ui_json["geoh5"] = workspace
    ui_json["workspace_geoh5"] = workspace.h5file
    ui_json["points"] = templates.object_parameter(
        value=str(original_points.uid),
    )
    in_file = InputFile(ui_json=ui_json)

    workspace2 = Workspace()

    in_file.copy_relatives(workspace2)

    # dry run for coverage
    InputFile().copy_relatives(workspace2)

    copied_points = workspace2.get_entity("Points_A")[0]

    compare_entities(copied_points, original_points, ignore=["_property_groups"])


def test_copy_uijson(tmp_path):
    workspace = get_workspace(tmp_path)
    original_points = workspace.get_entity("Points_A")[0]

    ui_json = deepcopy(default_ui_json)
    ui_json["geoh5"] = workspace
    ui_json["points"] = templates.object_parameter(
        value=str(original_points.uid),
    )
    in_file = InputFile(ui_json=ui_json)

    # same workspace
    copied_in_file = in_file.copy(title="Copied")
    assert copied_in_file.geoh5.h5file == workspace.h5file
    assert copied_in_file.data["title"] == "Copied"

    # with new workspace
    h5file_path_2 = tmp_path / "copied.geoh5"
    new_in_file = in_file.copy(geoh5=h5file_path_2)

    with Workspace(h5file_path_2) as workspace_2:
        assert str(new_in_file.geoh5.h5file)[-12:] == "copied.geoh5"
        assert workspace_2.get_entity("Points_A")[0].name == "Points_A"

    with pytest.raises(FileExistsError, match="The specified geoh5"):
        in_file.copy(geoh5=h5file_path_2)

    with pytest.raises(ValueError, match="InputFile must have a ui_json"):
        InputFile().copy()
