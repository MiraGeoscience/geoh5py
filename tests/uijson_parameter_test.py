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
from geoh5py.ui_json.templates import BaseParameter, FormParameter, StringParameter
from geoh5py.ui_json.validation import Validations


def test_base_parameter():
    param = BaseParameter("param", "nogood", Validations({"types": [int, float]}))
    with pytest.raises(TypeValidationError):
        param.validate()


def test_form_parameter():
    # Properly create a form parameter.
    param = FormParameter(
        "param",
        {"label": "my param", "value": "goodvalue"},
        Validations({"types": [str]}),
    )
    # Confirm that validations of the value member are postponed.
    param = FormParameter(
        "param",
        {"label": "my param", "value": "badvalue"},
        Validations({"types": [int]}),
    )
    # Catch invalid form member.
    with pytest.raises(TypeValidationError):
        param = FormParameter(
            "param",
            {"label": "my param", "value": "goodvalue", "optional": "whoops"},
            Validations({"types": [str]}),
        )
    # Catch incomplete form.
    with pytest.raises(UIJsonFormatError):
        param = FormParameter(
            "param", {"value": "goodvalue"}, Validations({"types": [str]})
        )

    # with pytest.raises(AggregateValidationError):
    #     param.label.value = None
    #     param.validate()
    # with pytest.raises(TypeValidationError):
    #     param.label.value = "nogood param"
    #     param.enabled.value = "notabool"
    #     param.validate()
    #
    # param = FormParameter(
    #     "param",
    #     "good param",
    #     "good",
    #     {"types": [str]},
    #     form={"label": "good param", "value": "good", "optional": True, "enabled": False}
    # )
    # assert True


def test_string_parameter():
    param = StringParameter(
        "inversion_type",
        "gravity",
        Validations({"required": True, "types": [str], "values": ["gravity"]}),
    )
    assert True
