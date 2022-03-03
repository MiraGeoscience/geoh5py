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

from copy import deepcopy

from geoh5py.ui_json import templates
from geoh5py.ui_json.constants import default_ui_json
from geoh5py.ui_json.utils import (
    collect,
    group_enabled,
    group_optional,
    is_form,
    is_uijson,
    optional_type,
    set_enabled,
    truth,
)


def test_dict_mapper():
    pass


def test_flatten():
    pass


def test_collect():
    d_u_j = deepcopy(default_ui_json)
    d_u_j["string_parameter"] = templates.string_parameter(optional="enabled")
    d_u_j["float_parameter"] = templates.float_parameter(optional="disabled")
    d_u_j["integer_parameter"] = templates.integer_parameter(optional="enabled")
    enabled_params = collect(d_u_j, "enabled", value=True)
    assert len(enabled_params) == 2
    assert all(
        k in enabled_params.keys() for k in ["string_parameter", "integer_parameter"]
    )
    tooltip_params = collect(d_u_j, "tooltip")
    assert len(tooltip_params) == 1
    assert "run_command_boolean" in tooltip_params.keys()


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


def test_optional_type():
    ui_json = deepcopy(default_ui_json)
    ui_json["string_parameter"] = templates.string_parameter()
    ui_json["string_parameter"]["group"] = "test"
    ui_json["string_parameter"]["groupOptional"] = True
    ui_json["float_parameter"] = templates.float_parameter()
    ui_json["float_parameter"]["group"] = "test"
    ui_json["integer_parameter"] = templates.integer_parameter()
    ui_json["integer_parameter"]["group"] = "test"
    ui_json["other_float_parameter"] = templates.float_parameter(optional="enabled")
    assert optional_type(ui_json, "integer_parameter")
    ui_json["string_parameter"]["groupOptional"] = False
    assert not optional_type(ui_json, "float_parameter")
    assert optional_type(ui_json, "other_float_parameter")


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

    # set_enabled[""]

    # If parameter is in group and groupOptional: True then disable
    # the parameter containing the groupOptional member.
    set_enabled(ui_json, "float_parameter", False)
    assert not ui_json["string_parameter"]["enabled"]
    assert ui_json["float_parameter"]["enabled"]
    set_enabled(ui_json, "float_parameter", True)
    assert ui_json["float_parameter"]["enabled"]
    ui_json["string_parameter"].pop("groupOptional")

    # Remove the groupOptional member and check that set_enabled
    # Affects the enabled status of the calling parameter
    set_enabled(ui_json, "float_parameter", False)
    assert not ui_json["float_parameter"]["enabled"]
    assert ui_json["string_parameter"]["enabled"]
    set_enabled(ui_json, "float_parameter", True)
    assert ui_json["float_parameter"]["enabled"]


def test_truth():
    pass


def test_is_uijson():
    pass


def test_is_form_test():
    pass
