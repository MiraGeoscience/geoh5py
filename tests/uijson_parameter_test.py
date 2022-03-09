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

from geoh5py.shared.exceptions import TypeValidationError
from geoh5py.ui_json.exceptions import UIJsonFormatError
from geoh5py.ui_json.templates import (
    Parameter,
    FormParameter,
    StringParameter,
    FloatParameter,
    DataValueParameter,
    UIJson,
)
from geoh5py.ui_json.validation import Validations


def test_parameter():
    param = Parameter("param", "nogood", {"types": [int, float]})
    with pytest.raises(TypeValidationError):
        param.validate()


def test_form_parameter():
    # Properly create a form parameter.
    param = FormParameter(
        "param",
        {"label": "my param", "value": "goodvalue"},
        {"types": [str]},
    )
    # Confirm that validations of the value member are postponed.
    param = FormParameter(
        "param",
        {"label": "my param", "value": "badvalue"},
        {"types": [int]},
    )
    # Catch invalid form member.
    with pytest.raises(TypeValidationError):
        param = FormParameter(
            "param",
            {"label": "my param", "value": "goodvalue", "optional": "whoops"},
            {"types": [str]},
        )
    # Catch incomplete form.
    with pytest.raises(UIJsonFormatError):
        param = FormParameter("param", {"value": "goodvalue"}, {"types": [str]})


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
        {}
    )
    assert param.value == 1.0
    param.isValue.value = False
    assert param.value is None


def test_uijson_identify():
    assert UIJson.identify({"min": 2}) == FloatParameter


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


def test_string_parameter():
    param = StringParameter(
        "inversion_type",
        "gravity",
        Validations({"required": True, "types": [str], "values": ["gravity"]}),
    )
    assert True
