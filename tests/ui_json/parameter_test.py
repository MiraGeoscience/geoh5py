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


import pytest

from geoh5py.shared.exceptions import TypeValidationError, UUIDValidationError
from geoh5py.ui_json.enforcers import TypeEnforcer, ValueEnforcer
from geoh5py.ui_json.parameters import (
    BoolParameter,
    FloatParameter,
    IntegerParameter,
    Parameter,
    StringListParameter,
    StringParameter,
    UUIDParameter,
)
from geoh5py.ui_json.validation import Validations


def test_no_overwriting_class_enforcers():
    param = StringParameter(
        "param",
        validations=Validations(
            "param", [TypeEnforcer(int), ValueEnforcer(["onlythis"])]
        ),
    )
    assert TypeEnforcer(str) in param.validations.enforcers
    assert ValueEnforcer(["onlythis"]) in param.validations.enforcers


def test_skip_validation_on_none_value():
    validations = Validations("my_param", [TypeEnforcer(str)])
    param = Parameter("my_param", None, validations=validations)
    assert param.value is None


def test_parameter_validations_on_construction():
    validations = Validations("my_param", [TypeEnforcer(str)])
    _ = Parameter("my_param", "me", validations=validations)
    msg = "Type 'int' provided for 'my_param' is invalid. Must be: 'str'."
    with pytest.raises(TypeValidationError, match=msg):
        _ = Parameter("my_param", 1, validations=validations)


def test_parameter_validations_on_setting():
    validations = Validations("my_param", [TypeEnforcer(str)])
    param = Parameter("my_param", validations=validations)
    param.value = "me"
    msg = "Type 'int' provided for 'my_param' is invalid. Must be: 'str'."
    with pytest.raises(TypeValidationError, match=msg):
        param.value = 1


def test_raise_if_no_validations_on_validate():
    param = Parameter("my_param")
    with pytest.raises(AttributeError, match="Must set validations"):
        param.validate()


def test_parameter_str_representation():
    param = Parameter("my_param")
    assert str(param) == "<Parameter> : 'my_param' -> None"


def test_string_parameter_type_validation():
    param = StringParameter("my_param")
    msg = "Type 'int' provided for 'my_param' is invalid. " "Must be: 'str'."
    with pytest.raises(TypeValidationError, match=msg):
        param.value = 1


def test_string_parameter_optional_validations():
    param = StringParameter("my_param", optional=True)
    param.enforcers == [TypeEnforcer([str, type(None)])]
    param.value = None
    param.value = "this is ok"
    msg = (
        "Type 'int' provided for 'my_param' is invalid. "
        "Must be one of: 'str', 'NoneType'."
    )
    with pytest.raises(TypeValidationError, match=msg):
        param.value = 1


def test_integer_parameter_type_validation():
    param = IntegerParameter("my_param")
    msg = "Type 'str' provided for 'my_param' is invalid. " "Must be: 'int'."
    with pytest.raises(TypeValidationError, match=msg):
        param.value = "1"


def test_float_parameter_type_validation():
    param = FloatParameter("my_param")
    msg = "Type 'int' provided for 'my_param' is invalid. " "Must be: 'float'."
    with pytest.raises(TypeValidationError, match=msg):
        param.value = 1


def test_bool_parameter_type_validation():
    param = BoolParameter("my_param")
    msg = "Type 'str' provided for 'my_param' is invalid. " "Must be: 'bool'."
    with pytest.raises(TypeValidationError, match=msg):
        param.value = "butwhy?"


def test_uuid_parameter_type_validation():
    param = UUIDParameter("my_param")
    msg = "Parameter 'my_param' with value 'notauuid' is not a valid uuid string."
    with pytest.raises(UUIDValidationError, match=msg):
        param.value = "notauuid"


def test_string_list_parameter_type_validation():
    param = StringListParameter("my_param")
    param.value = "this is ok"
    param.value = ["this", "is", "also", "ok"]
    msg = (
        "Type 'int' provided for 'my_param' is invalid. "
        "Must be one of: 'list', 'str'."
    )
    with pytest.raises(TypeValidationError, match=msg):
        param.value = 1
