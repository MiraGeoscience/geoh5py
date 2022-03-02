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
        [k in enabled_params.keys() for k in ["string_parameter", "integer_parameter"]]
    )
    tooltip_params = collect(d_u_j, "tooltip")
    assert len(tooltip_params) == 1
    assert "run_command_boolean" in tooltip_params.keys()


def group_optional():
    d_u_j = deepcopy(default_ui_json)
    d_u_j["string_parameter"] = templates.string_parameter(optional="enabled")
    d_u_j["float_parameter"] = templates.float_parameter(optional="disabled")
    d_u_j["integer_parameter"] = templates.integer_parameter(optional="enabled")


def optional_type():
    pass


def set_enabled():
    pass


def truth():
    pass


def is_uijson():
    pass


def is_form_test():
    pass
