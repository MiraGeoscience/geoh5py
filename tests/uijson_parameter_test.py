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

import pytest

from geoh5py.shared.exceptions import (
    AggregateValidationError,
    TypeValidationError,
    UIJsonFormatError,
)
from geoh5py.ui_json.templates import (
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
    UIJson,
)
from geoh5py.ui_json.validation import Validations


def test_parameter():
    param = Parameter("param", "nogood", {"types": [int, float]})
    with pytest.raises(TypeValidationError):
        param.validate()
    param = Parameter(
        "param", "nogood", {"values": ["onlythis"], "types": [int, float]}
    )
    with pytest.raises(AggregateValidationError):
        param.validate()


def test_form_parameter_validate():

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
        {"required": True, "types": [str], "values": ["gravity"]},
    )
    assert True


def test_something():
    test = Parameter("me", 2, {})
    test.__str__()
    assert True


def test_float_parameter():
    # FloatFormParameter should add the "types": [float] validations
    # and min/max form_validations by default.
    param = FloatParameter(
        "param", {"label": "my param", "value": 1}, {"required": True}
    )
    assert all(k in param.validations for k in ["types", "required"])
    assert all(k in param.form_validations for k in ["min", "max"])


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
        {},
    )
    assert param.value == 1.0
    param.isValue.value = False
    assert param.value is None


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
    assert UIJson.identify({"value": "lskddk"}) == FormParameter
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
    values = ui_json.values
    assert all()
