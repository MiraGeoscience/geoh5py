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

import uuid

import pytest

from geoh5py.shared.exceptions import (
    AggregateValidationError,
    TypeValidationError,
    ValueValidationError,
)
from geoh5py.shared.validators import ValueValidator
from geoh5py.ui_json.parameters import (
    BoolParameter,
    ChoiceStringParameter,
    DataParameter,
    DataValueParameter,
    FileParameter,
    FloatParameter,
    FormParameter,
    IntegerParameter,
    ObjectParameter,
    Parameter,
    StringParameter,
)
from geoh5py.ui_json.ui_json import UIJson
from geoh5py.ui_json.validation import Validations

# pylint: disable=protected-access, invalid-name


def test_parameter():
    # Parameters can be instantiated with just a name
    param = Parameter("param")
    assert param.name == "param"
    assert param.value is None
    assert not param.validations

    # Parameters can be instantiated with a name and value
    param = Parameter("param", "test")
    assert param.name == "param"
    assert param.value == "test"
    assert not param.validations

    # Validations are empty by default and cannot validate until set
    with pytest.raises(AttributeError, match="Must set validations"):
        param.validate()

    # set validations and validate should pass
    param.validations.update({"values": ["test"]})
    assert isinstance(param.validations, Validations)
    assert param.validations.validators == [ValueValidator]
    param.validate()

    # validations setter promotes dictionaries to Validations
    param.validations = {"types": [str]}
    assert isinstance(param.validations, Validations)
    assert "types" in param.validations
    param.validate()

    # updates to param.validations are reflected in validate calls
    param.validations.update({"values": ["nogood"]})
    with pytest.raises(ValueValidationError):
        param.validate()

    # setting None replaces value
    param.validations = None
    with pytest.raises(AttributeError, match="Must set validations"):
        param.validate()

    # Bad type triggers TypeValidationError
    with pytest.raises(TypeValidationError):
        param = Parameter("param", "nogood", {"types": [int, float]})

    # Bad type and value triggers AggregateValidationError
    with pytest.raises(AggregateValidationError):
        param = Parameter(
            "param", "nogood", {"values": ["onlythis"], "types": [int, float]}
        )

    assert str(param) == "<Parameter> : 'param' -> test"


def test_validation_update():
    param = Parameter("param", "test", {"types": [str], "values": ["test", "me"]})
    param.validations = dict(param.validations, **{"values": ["you"], "required": True})
    assert len(param.validations) == 3
    assert len(param._validations.validators) == 3
    assert param.validations["values"] == ["you"]
    assert param.validations["required"]
    assert param.validations["types"] == [str]


def test_form_parameter_roundtrip():
    form = {"label": "my param", "value": 1, "extra": "stuff"}
    param = FormParameter("param", **form)
    assert param.name == "param"
    assert param.label == "my param"
    assert param.validations == {}
    assert not hasattr(param, "extra")
    assert param._extra_members["extra"] == "stuff"
    assert all(hasattr(param, k) for k in param.valid_members)
    assert param.form == form


def test_form_parameter_validate():
    # Blank initialization shouldn't crash, register should pass
    # with valid data and fail with invalid data
    param = FormParameter("param")
    param.register({"label": "my param", "value": 1})
    with pytest.raises(
        TypeValidationError, match="Type 'str' provided for 'optional' is invalid"
    ):
        param.register({"optional": "whoops"})

    # should raise aggregated error with register of bad value and member
    param.validations = {"types": [str]}
    assert param._value.validations == {"types": [str]}
    with pytest.raises(
        AggregateValidationError, match="Validation of 'param' collected 2 errors:"
    ):
        param.register({"optional": "whoops", "value": 1})

    # Form validations run on instantiation should pass
    param = FormParameter.from_dict(
        "param", {"label": "my param", "value": 1}, validations={"types": [int]}
    )
    assert param.name == "param"
    assert len(param.form) == 2
    assert all(k in param.form for k in ["label", "value"])
    assert param.label == "my param"
    assert param.value == 1
    assert all(hasattr(param, k) for k in param.valid_members)

    # Form validation should fail when form is invalid
    with pytest.raises(
        TypeValidationError, match="Type 'str' provided for 'enabled' is invalid"
    ):
        param.enabled = "uh-oh"

    # Multiple form (member and value) errors are aggregated
    with pytest.raises(
        AggregateValidationError, match="Validation of 'param' collected 2 errors:"
    ):
        param = FormParameter.from_dict(
            "param",
            {"label": "my param", "value": 1, "optional": "whoops"},
            {"types": [str]},
        )


def test_string_parameter():
    param = StringParameter.from_dict(
        "inversion_type",
        {"label": "inversion type", "value": "gravity"},
    )
    assert len(param.validations) == 1
    assert "types" in param.validations
    assert str(param) == "<StringParameter> : 'inversion_type' -> gravity"


def test_bool_parameter():
    param = BoolParameter.from_dict(
        "gz_channel_bool",
        {"label": "gz", "value": True},
    )
    assert len(param.validations) == 1
    assert "types" in param.validations
    assert str(param) == "<BoolParameter> : 'gz_channel_bool' -> True"


def test_float_parameter():
    # FloatFormParameter should add the "types": [float] validations
    # and min/max form_validations by default.
    param = FloatParameter.from_dict(
        "param", {"label": "my param", "value": 1.0}, {"required": True}
    )
    assert all(k in param.validations for k in ["types", "required"])
    assert all(
        k in param.form_validations for k in ["min", "max", "precision", "line_edit"]
    )
    assert str(param) == "<FloatParameter> : 'param' -> 1.0"


def test_choice_string_parameter():
    param = ChoiceStringParameter.from_dict(
        "param", {"label": "methods", "choiceList": ["cg", "ssor"], "value": "cg"}
    )
    assert all(k in param.validations for k in ["types"])
    assert "choice_list" in param.form_validations
    reqd = ["label", "value", "choice_list"]
    assert all(k in param.required for k in reqd)
    assert str(param) == "<ChoiceStringParameter> : 'param' -> cg"


def test_file_parameter():
    param = FileParameter.from_dict(
        "param",
        {
            "label": "path",
            "fileDescription": "comma separated values",
            "fileType": "csv",
            "value": "test.csv",
        },
        validations={"required": True},
    )
    assert all(k in param.validations for k in ["types", "required"])
    assert all(
        k in param.form_validations
        for k in ["file_description", "file_type", "file_multi"]
    )
    reqd = ["label", "value", "file_description", "file_type"]
    assert all(k in param.required for k in reqd)
    assert str(param) == "<FileParameter> : 'param' -> test.csv"


def test_object_parameter():
    uid = uuid.uuid4()
    param = ObjectParameter.from_dict(
        "param",
        {
            "label": "mesh",
            "meshType": "{202C5DB1-A56D-4004-9CAD-BAAFD8899406}",
            "value": uid,
        },
    )
    assert all(k in param.validations for k in ["types"])
    assert "mesh_type" in param.form_validations
    reqd = ["label", "value", "mesh_type"]
    assert all(k in param.required for k in reqd)
    assert str(param) == f"<ObjectParameter> : 'param' -> {uid}"


def test_data_parameter():
    uid = uuid.uuid4()
    param = DataParameter.from_dict(
        "param",
        {
            "label": "gz_channel",
            "parent": uid,
            "association": "Vertex",
            "dataType": "Float",
        },
    )
    assert all(k in param.validations for k in ["types"])
    assert all(
        k in param.form_validations
        for k in ["parent", "association", "data_type", "data_group_type"]
    )
    reqd = ["label", "value", "parent", "association", "data_type"]
    assert all(k in param.required for k in reqd)
    assert str(param) == "<DataParameter> : 'param' -> None"


def test_data_value_parameter():
    param = DataValueParameter.from_dict(
        "param",
        {
            "association": "Vertex",
            "dataType": "Float",
            "isValue": True,
            "property": "",
            "parent": "other_param",
            "label": "my param",
            "value": 1.0,
        },
    )
    assert param.value == 1.0
    param.is_value = False
    assert param.value == ""
    reqd = ["label", "value", "parent", "association", "data_type", "is_value"]
    assert all(k in param.required for k in reqd)

    assert "types" in param.validations
    assert all(
        k in param.form_validations
        for k in ["parent", "association", "data_type", "is_value", "property"]
    )

    # incomplete form results in UIJsonFormatError
    with pytest.raises(
        AggregateValidationError, match="Validation of 'param' collected 5 errors:"
    ):
        param = DataValueParameter.from_dict(
            "param",
            {
                "label": "my param",
                "value": 1.0,
            },
        )
    assert str(param) == "<DataValueParameter> : 'param' -> "


def test_parameter_class():
    assert UIJson._parameter_class({"label": "lsdfkj"}) == FormParameter
    assert UIJson._parameter_class({"choiceList": ["lsdkfj"]}) == ChoiceStringParameter
    assert UIJson._parameter_class({"fileDescription": "lskdjf"}) == FileParameter
    assert UIJson._parameter_class({"fileType": "sldkjf"}) == FileParameter
    assert UIJson._parameter_class({"meshType": "lsdkjf"}) == ObjectParameter
    assert UIJson._parameter_class({"dataGroupType": "Multi-element"}) == DataParameter
    assert UIJson._parameter_class({"isValue": True}) == DataValueParameter
    assert UIJson._parameter_class({"property": "lskdjf"}) == DataValueParameter


def test_possible_parameter_classes():
    possibilities = UIJson._possible_parameter_classes({"label": "test", "value": 2})
    assert all(k in FormParameter.__subclasses__() for k in possibilities)
    possibilities = UIJson._possible_parameter_classes({"min"})
    assert all(k in [IntegerParameter, FloatParameter] for k in possibilities)
    possibilities = UIJson._possible_parameter_classes({"max"})
    assert all(k in [IntegerParameter, FloatParameter] for k in possibilities)
    possibilities = UIJson._possible_parameter_classes({"precision"})
    assert all(k in [FloatParameter] for k in possibilities)
    possibilities = UIJson._possible_parameter_classes({"lineEdit"})
    assert all(k in [FloatParameter] for k in possibilities)
    possibilities = UIJson._possible_parameter_classes({"fileMulti"})
    assert all(k in [FileParameter] for k in possibilities)
    possibilities = UIJson._possible_parameter_classes({"parent"})
    assert all(k in [DataParameter, DataValueParameter] for k in possibilities)
    possibilities = UIJson._possible_parameter_classes({"association"})
    assert all(k in [DataParameter, DataValueParameter] for k in possibilities)
    possibilities = UIJson._possible_parameter_classes({"dataType"})
    assert all(k in [DataParameter, DataValueParameter] for k in possibilities)


def test_identify():
    assert UIJson.identify({"label": "test", "value": "lsdkjf"}) == StringParameter
    assert UIJson.identify({"label": "test", "value": 2}) == IntegerParameter
    assert UIJson.identify({"label": "test", "value": 2.0}) == FloatParameter
    assert UIJson.identify({"precision": 2}) == FloatParameter
    assert UIJson.identify({"lineEdit": True}) == FloatParameter
    assert (
        UIJson.identify(
            {
                "choiceList": [
                    2,
                ]
            }
        )
        == ChoiceStringParameter
    )
    assert UIJson.identify({"fileDescription": "lskdjf"}) == FileParameter
    assert UIJson.identify({"fileType": "lskdjf"}) == FileParameter
    assert UIJson.identify({"fileMulti": True}) == FileParameter
    assert UIJson.identify({"meshType": "lsdkfj"}) == ObjectParameter
    assert UIJson.identify({"parent": "sldkfj", "dataType": "Vertex"}) == DataParameter
    assert (
        UIJson.identify({"association": "Vertex", "dataType": "Vertex"})
        == DataParameter
    )
    assert UIJson.identify({"dataType": "Float"}) == DataParameter
    assert UIJson.identify({"dataGroupType": "Multi-element"}) == DataParameter
    assert UIJson.identify({"isValue": True}) == DataValueParameter
    assert UIJson.identify({"property": None}) == DataValueParameter


def test_uijson():
    parameters = {
        "param_1": FormParameter.from_dict(
            "param_1",
            {"label": "first parameter", "value": "toocool"},
            {"types": [str]},
        ),
        "param_2": StringParameter.from_dict(
            "param_2", {"label": "second parameter", "value": "ohyeah"}
        ),
        "param_3": Parameter("param_3", 2, {"types": [int]}),
        "param_4": {"label": "fourth parameter", "value": 2},
    }
    ui_json = UIJson(parameters)
    p1 = ui_json.parameters["param_1"]
    assert isinstance(p1, FormParameter)
    assert p1.name == "param_1"
    assert p1.label == "first parameter"
    assert p1.value == "toocool"
    assert p1.validations == {"types": [str]}
    p2 = ui_json.parameters["param_2"]
    assert isinstance(p2, StringParameter)
    assert p2.name == "param_2"
    assert p2.label == "second parameter"
    assert p2.value == "ohyeah"
    assert p2.validations == {"types": [str]}
    p3 = ui_json.parameters["param_3"]
    assert isinstance(p3, Parameter)
    assert p3.name == "param_3"
    assert p3.value == 2
    assert p3.validations == {"types": [int]}
    p4 = ui_json.parameters["param_4"]
    assert isinstance(p4, IntegerParameter)
    assert p4.name == "param_4"
    assert p4.label == "fourth parameter"
    assert p4.value == 2
    assert p4.validations == {"types": [int]}
    ui_json.validate()
