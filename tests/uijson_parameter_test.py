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
    UIJsonFormatError,
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


def test_parameter():
    # Validations can be instantiated with a name and value
    param = Parameter("param", "test")
    assert param.name == "param"
    assert param.value == "test"
    assert param.validations is None

    # validations default to None but cannot validate until set
    with pytest.raises(AttributeError, match="Must set validations"):
        param.validate()

    # set validations and validate should pass
    param.validations = {"values": ["test"]}
    assert isinstance(param._validations, Validations)
    assert param._validations.validators == [ValueValidator]
    assert isinstance(param.validations, dict)
    param.validate()

    # validations setter overwrites existing validations
    param.validations = {"types": [str]}
    assert isinstance(param._validations, Validations)
    assert isinstance(param.validations, dict)
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
    param = Parameter("param", "nogood", {"types": [int, float]})
    with pytest.raises(TypeValidationError):
        param.validate()

    # Bad type and value triggers AggregateValidationError
    param = Parameter(
        "param", "nogood", {"values": ["onlythis"], "types": [int, float]}
    )
    with pytest.raises(AggregateValidationError):
        param.validate()

    assert param.__str__() == "<Parameter> : 'param' -> nogood"


def test_validation_update():
    param = Parameter("param", "test", {"types": [str], "values": ["test", "me"]})
    param.validations = dict(param.validations, **{"values": ["you"], "required": True})
    assert len(param.validations) == 3
    assert len(param._validations.validators) == 3
    assert param.validations["values"] == ["you"]
    assert param.validations["required"]
    assert param.validations["types"] == [str]


def test_form_parameter_validate():
    param = FormParameter(
        "param", {"label": "my param", "value": 1}, validations={"types": [int]}
    )
    assert param.name == "param"
    assert len(param.form) == 2
    assert all(k in param.form for k in ["label", "value"])
    assert param.label.value == "my param"
    assert param.value == 1
    assert all(
        [
            isinstance(getattr(param, k), Parameter)
            for k in FormParameter.valid_members
            if k != "value"
        ]
    )
    param.validate("form")
    param.validate("value")
    param.validate("all")

    param = FormParameter(
        "param",
        {"label": "my param", "value": 1, "optional": "whoops"},
        {"types": [str]},
    )

    # Test form validations
    with pytest.raises(UIJsonFormatError) as excinfo:
        param.validate()
    assert all(n in str(excinfo.value) for n in ["'param'", "'str'", "'optional'"])

    # Test value validation
    with pytest.raises(TypeValidationError) as excinfo:
        param.validate(level="value")
    assert all(n in str(excinfo.value) for n in ["'int'", "'value'", "'str'"])

    param.validations["values"] = [2, 3]
    # Aggregate form and value validations
    with pytest.raises(AggregateValidationError) as excinfo:
        param.validate(level="all")
    assert all(
        n in str(excinfo.value)
        for n in ["'param'", "2 error", "0. Type 'int'", "1. Invalid UIJson"]
    )


def test_string_parameter():
    param = StringParameter(
        "inversion_type",
        {"label": "inversion type", "value": "gravity"},
    )
    assert len(param.validations) == 1
    assert "types" in param.validations
    param.validate("all")


def test_bool_parameter():
    param = BoolParameter(
        "gz_channel_bool",
        {"label": "gz", "value": True},
    )
    assert len(param.validations) == 1
    assert "types" in param.validations
    param.validate("all")


def test_float_parameter():
    # FloatFormParameter should add the "types": [float] validations
    # and min/max form_validations by default.
    param = FloatParameter(
        "param", {"label": "my param", "value": 1}, {"required": True}
    )
    assert all(k in param.validations for k in ["types", "required"])
    assert all(
        k in param.form_validations for k in ["min", "max", "precision", "lineEdit"]
    )
    param.validate("all")


def test_choice_string_parameter():
    param = ChoiceStringParameter(
        "param", {"label": "methods", "choiceList": ["cg", "ssor"], "value": "cg"}
    )
    assert all(k in param.validations for k in ["types"])
    assert "choiceList" in param.form_validations
    param.validate("all")


def test_file_parameter():
    param = FileParameter(
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
        for k in ["fileDescription", "fileType", "fileMulti"]
    )
    param.validate("all")


def test_object_parameter():
    param = ObjectParameter(
        "param",
        {
            "label": "mesh",
            "meshType": "{202C5DB1-A56D-4004-9CAD-BAAFD8899406}",
            "value": uuid.uuid4(),
        },
    )
    assert all(k in param.validations for k in ["types"])
    assert "meshType" in param.form_validations
    param.validate("all")


def test_data_parameter():
    param = DataParameter(
        "param",
        {
            "label": "gz_channel",
            "parent": uuid.uuid4(),
            "association": "Vertex",
            "dataType": "Float",
        },
    )
    assert all(k in param.validations for k in ["types"])
    assert all(
        k in param.form_validations
        for k in ["parent", "association", "dataType", "dataGroupType"]
    )
    param.validate("all")


def test_data_value_parameter():
    param = DataValueParameter(
        "param",
        {
            "association": "Vertex",
            "dataType": "Float",
            "isValue": True,
            "property": None,
            "parent": "other_param",
            "label": "my param",
            "value": 1.0,
        },
    )
    assert param.value == 1.0
    param.isValue.value = False
    assert param.value is None

    assert "types" in param.validations
    assert all(
        k in param.form_validations
        for k in ["parent", "association", "dataType", "isValue", "property"]
    )
    # param.validate("all")

    # incomplete form results in UIJsonFormatError
    param = DataValueParameter(
        "param",
        {
            "label": "my param",
            "value": 1.0,
        },
    )
    with pytest.raises(UIJsonFormatError):
        param.validate()


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
        "param_1": FormParameter(
            "param_1",
            {"label": "first parameter", "value": "toocool"},
            {"types": [str]},
        ),
        "param_2": FormParameter(
            "param_2", {"label": "second parameter", "value": 2}, {"types": [int]}
        ),
        "param_3": Parameter("param_3", 2, {"types": [int]}),
    }
    ui_json = UIJson(parameters)
    ui_json.validate()
