# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2025-2026 Mira Geoscience Ltd.                                '
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

import uuid
from copy import deepcopy

import pytest

from geoh5py.shared.utils import dict_mapper
from geoh5py.ui_json import templates
from geoh5py.ui_json.constants import default_ui_json
from geoh5py.ui_json.utils import (
    collect,
    flatten,
    group_enabled,
    group_optional,
    is_form,
    is_uijson,
    requires_value,
    set_enabled,
    truth,
)
from geoh5py.ui_json.validation import InputValidation


def test_dict_mapper():
    tdict = {"key1": {"key2": {"key3": "yargh"}}}

    def func(x):
        return x[:-1] if x == "yargh" else x

    for key, value in tdict.items():
        value = dict_mapper(value, [func])
        tdict[key] = value
    assert tdict["key1"]["key2"]["key3"] == "yarg"


def test_flatten():
    ui_json = deepcopy(default_ui_json)
    ui_json["string_parameter"] = templates.string_parameter(
        optional="enabled", value="hello"
    )
    ui_json["float_parameter"] = templates.float_parameter(
        optional="disabled", value=1.0
    )
    ui_json["some_other_parameter"] = templates.data_value_parameter(
        is_value=False, value=2.0, prop="{some_uuid_string}"
    )
    data = flatten(ui_json)
    assert data["string_parameter"] == "hello"
    assert data["float_parameter"] is None
    assert data["some_other_parameter"] == "{some_uuid_string}"


def test_collect():
    ui_json = deepcopy(default_ui_json)
    ui_json["string_parameter"] = templates.string_parameter(optional="enabled")
    ui_json["string_parameter"]["tooltip"] = "Bla bla bla"
    ui_json["float_parameter"] = templates.float_parameter(optional="disabled")
    ui_json["integer_parameter"] = templates.integer_parameter(optional="enabled")
    enabled_params = collect(ui_json, "enabled", value=True)
    assert len(enabled_params) == 2
    assert all(k in enabled_params for k in ["string_parameter", "integer_parameter"])
    tooltip_params = collect(ui_json, "tooltip")
    assert len(tooltip_params) == 1
    assert "string_parameter" in tooltip_params


def test_group_optional():
    ui_json = deepcopy(default_ui_json)
    ui_json["string_parameter"] = templates.string_parameter(optional="enabled")
    ui_json["string_parameter"]["group"] = "test"
    ui_json["string_parameter"]["groupOptional"] = True
    ui_json["float_parameter"] = templates.float_parameter(optional="enabled")
    ui_json["float_parameter"]["group"] = "test"
    ui_json["integer_parameter"] = templates.integer_parameter(optional="enabled")
    ui_json["integer_parameter"]["group"] = "test"
    assert group_optional(ui_json, "test")
    ui_json["string_parameter"].pop("groupOptional")
    assert not group_optional(ui_json, "test")
    ui_json["string_parameter"]["groupOptional"] = False
    assert not group_optional(ui_json, "test")


def test_requires_value():
    # check default parameters
    ui_json = deepcopy(default_ui_json)
    ui_json["string_parameter"] = templates.string_parameter()
    ui_json["string_parameter"]["group"] = "test"
    ui_json["string_parameter"]["enabled"] = True
    ui_json["string_parameter"]["groupOptional"] = (
        False  # equivalent to not having groupOptional
    )
    ui_json["float_parameter"] = templates.float_parameter(optional="enabled")
    ui_json["float_parameter"]["group"] = "test"
    ui_json["integer_parameter"] = templates.integer_parameter()
    ui_json["integer_parameter"]["group"] = "test"
    assert requires_value(ui_json, "string_parameter")
    assert requires_value(ui_json, "float_parameter")
    assert requires_value(ui_json, "integer_parameter")

    # Check the groupOptional behaviour
    ui_json["string_parameter"].pop("enabled")
    ui_json["string_parameter"]["groupOptional"] = True
    assert requires_value(ui_json, "string_parameter")
    assert requires_value(ui_json, "float_parameter")
    assert requires_value(ui_json, "integer_parameter")

    ui_json["string_parameter"]["enabled"] = False
    assert not requires_value(ui_json, "string_parameter")
    assert not requires_value(ui_json, "float_parameter")
    assert not requires_value(ui_json, "integer_parameter")

    # Check the optional dependency behaviour
    ui_json.pop("string_parameter")
    ui_json["float_parameter"]["enabled"] = True
    ui_json["integer_parameter"]["dependency"] = "float_parameter"
    ui_json["integer_parameter"]["dependencyType"] = "enabled"
    assert requires_value(ui_json, "float_parameter")
    assert requires_value(ui_json, "integer_parameter")

    ui_json["float_parameter"]["enabled"] = False
    assert not requires_value(ui_json, "float_parameter")
    assert not requires_value(ui_json, "integer_parameter")

    ui_json["integer_parameter"]["dependencyType"] = "disabled"
    assert not requires_value(ui_json, "float_parameter")
    assert requires_value(ui_json, "integer_parameter")

    ui_json["float_parameter"]["enabled"] = True
    assert requires_value(ui_json, "float_parameter")
    assert not requires_value(ui_json, "integer_parameter")


def test_bool_required_value():
    ui_json = deepcopy(default_ui_json)
    ui_json["bool_parameter"] = templates.bool_parameter()
    ui_json["integer_parameter"] = templates.integer_parameter()
    ui_json["bool_parameter"]["value"] = True
    ui_json["integer_parameter"]["dependency"] = "bool_parameter"
    ui_json["integer_parameter"]["dependencyType"] = "enabled"
    assert requires_value(ui_json, "bool_parameter")
    assert requires_value(ui_json, "integer_parameter")
    ui_json["bool_parameter"]["value"] = False
    assert requires_value(ui_json, "bool_parameter")
    assert not requires_value(ui_json, "integer_parameter")

    ui_json["integer_parameter"]["dependencyType"] = "disabled"
    assert requires_value(ui_json, "bool_parameter")
    assert requires_value(ui_json, "integer_parameter")
    ui_json["bool_parameter"]["value"] = True
    assert requires_value(ui_json, "bool_parameter")
    assert not requires_value(ui_json, "integer_parameter")

    # Check optional inside boolean dependency
    ui_json["integer_parameter"]["dependencyType"] = "enabled"
    ui_json["bool_parameter"]["value"] = True
    ui_json["integer_parameter"]["optional"] = True
    ui_json["integer_parameter"]["enabled"] = True
    assert requires_value(ui_json, "integer_parameter")
    ui_json["bool_parameter"]["value"] = False
    assert not requires_value(ui_json, "integer_parameter")
    ui_json["integer_parameter"]["enabled"] = False
    assert not requires_value(ui_json, "integer_parameter")
    ui_json["bool_parameter"]["value"] = True
    assert not requires_value(ui_json, "integer_parameter")

    ui_json["integer_parameter"]["dependencyType"] = "disabled"
    ui_json["bool_parameter"]["value"] = True
    ui_json["integer_parameter"]["enabled"] = True
    assert not requires_value(ui_json, "integer_parameter")
    ui_json["bool_parameter"]["value"] = False
    assert requires_value(ui_json, "integer_parameter")
    ui_json["integer_parameter"]["enabled"] = False
    assert not requires_value(ui_json, "integer_parameter")
    ui_json["bool_parameter"]["value"] = True
    assert not requires_value(ui_json, "integer_parameter")

    # Check the normal behaviour
    ui_json["other_parameter"] = templates.float_parameter()
    assert requires_value(ui_json, "other_parameter")


def test_group_enabled():
    ui_json = deepcopy(default_ui_json)
    ui_json["string_parameter"] = templates.string_parameter()
    ui_json["string_parameter"]["group"] = "test"
    ui_json["string_parameter"]["groupOptional"] = True
    ui_json["string_parameter"]["enabled"] = True
    ui_json["float_parameter"] = templates.float_parameter()
    ui_json["float_parameter"]["group"] = "test"
    ui_json["integer_parameter"] = templates.integer_parameter()
    ui_json["integer_parameter"]["group"] = "test"
    assert group_enabled(collect(ui_json, "group", "test"))
    ui_json["string_parameter"]["enabled"] = False
    assert not group_enabled(collect(ui_json, "group", "test"))


def test_set_enabled():
    ui_json = deepcopy(default_ui_json)
    ui_json["string_parameter"] = templates.string_parameter()
    ui_json["string_parameter"]["group"] = "test"
    ui_json["string_parameter"]["groupOptional"] = True
    ui_json["string_parameter"]["enabled"] = True
    ui_json["float_parameter"] = templates.float_parameter()
    ui_json["float_parameter"]["group"] = "test"
    ui_json["integer_parameter"] = templates.integer_parameter()
    ui_json["integer_parameter"]["group"] = "test"

    # If parameter is in group and groupOptional: True then disable
    # the parameter containing the groupOptional member.
    set_enabled(ui_json, "string_parameter", False)
    assert not ui_json["string_parameter"]["enabled"]
    assert not ui_json["float_parameter"]["enabled"]
    assert not ui_json["integer_parameter"]["enabled"]

    set_enabled(ui_json, "string_parameter", True)
    assert ui_json["string_parameter"]["enabled"]
    assert ui_json["float_parameter"]["enabled"]
    assert ui_json["integer_parameter"]["enabled"]

    # Remove the groupOptional member and check that set_enabled
    # Affects the enabled status of the calling parameter
    ui_json["string_parameter"].pop("groupOptional")
    with pytest.warns(UserWarning) as warn:
        set_enabled(ui_json, "float_parameter", False)

    assert (
        "Non-option parameter 'float_parameter' cannot be set to 'enabled' False "
    ) in str(warn[0])

    ui_json["float_parameter"]["optional"] = True
    set_enabled(ui_json, "float_parameter", False)
    assert not ui_json["float_parameter"]["enabled"]
    assert ui_json["string_parameter"]["enabled"]
    assert ui_json["integer_parameter"]["enabled"]

    set_enabled(ui_json, "float_parameter", True)
    assert ui_json["float_parameter"]["enabled"]


def test_truth():
    ui_json = deepcopy(default_ui_json)
    ui_json["string_parameter"] = templates.string_parameter(optional="disabled")
    assert truth(ui_json, "string_parameter", "optional")
    assert not truth(ui_json, "string_parameter", "enabled")
    assert not truth(ui_json, "string_parameter", "groupOptional")
    assert truth(ui_json, "string_parameter", "isValue")


def test_is_uijson():
    ui_json = deepcopy(default_ui_json)
    assert is_uijson(ui_json)
    ui_json.pop("title")
    assert not is_uijson(ui_json)


def test_is_form_test():
    ui_json = deepcopy(default_ui_json)
    ui_json["string_parameter"] = templates.string_parameter(optional="disabled")
    assert is_form(ui_json["string_parameter"])
    ui_json["string_parameter"].pop("value")
    assert not is_form(ui_json["string_parameter"])


def test_flatten_group_value():
    ui_json = deepcopy(default_ui_json)
    my_uuid = uuid.uuid4()

    ui_json["test"] = templates.drillhole_group_data(
        value=["test1"],
        group_value=my_uuid,
    )

    validators = InputValidation._validations_from_uijson(ui_json)
    assert validators["test"]["types"] == [dict]

    flat = flatten(ui_json)

    assert flat["test"] == {
        "groupValue": ui_json["test"]["groupValue"],
        "value": ["test1"],
    }

    ui_json = deepcopy(default_ui_json)
    ui_json["test"] = templates.drillhole_group_data(
        value=None,
        group_value=None,
        optional="enabled",
    )
    validators = InputValidation._validations_from_uijson(ui_json)
    assert validators["test"]["types"] == [dict]

    flat = flatten(ui_json)

    assert flat["test"] is None


def test_range_label():
    ui_json = deepcopy(default_ui_json)
    my_uuid = uuid.uuid4()
    ui_json["test"] = templates.range_label_template(
        value=[1, 2], property_=my_uuid, is_complement=False, optional="enabled"
    )

    validators = InputValidation._validations_from_uijson(ui_json)
    assert validators["test"]["types"] == [dict]

    flat = flatten(ui_json)

    assert flat["test"] == {"isComplement": False, "property": my_uuid, "value": [1, 2]}


def test_optional_error():
    with pytest.raises(ValueError, match="Unrecognized state option."):
        templates.optional_parameter("bidon")
