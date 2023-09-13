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

from geoh5py.shared.exceptions import (
    BaseValidationError,
    RequiredFormMemberValidationError,
    RequiredObjectDataValidationError,
    RequiredUIJsonParameterValidationError,
    RequiredWorkspaceObjectValidationError,
)


def test_forget_implement_message():
    class my_validation_error(BaseValidationError):
        pass

    err = my_validation_error()
    with pytest.raises(NotImplementedError):
        err.message("me", 1, [1, 2, 3])


def test_required_form_member_validation_error():
    msg = r"Form: 'test' is missing required member\(s\): \['member1', 'member2'\]."
    with pytest.raises(RequiredFormMemberValidationError, match=msg):
        raise RequiredFormMemberValidationError("test", ["member1", "member2"])


def test_required_ui_json_parameter_validation_error():
    msg = r"UIJson: 'test' is missing required parameter\(s\): \['param1', 'param2'\]."
    with pytest.raises(RequiredUIJsonParameterValidationError, match=msg):
        raise RequiredUIJsonParameterValidationError("test", ["param1", "param2"])


def test_required_workspace_object_validation_error():
    msg = r"Workspace: 'test' is missing required object\(s\): \['obj1', 'obj2'\]."
    with pytest.raises(RequiredWorkspaceObjectValidationError, match=msg):
        raise RequiredWorkspaceObjectValidationError("test", ["obj1", "obj2"])


def test_required_object_data_validation_error():
    msg = r"Object: 'test' is missing required data\(s\): \['data1', 'data2'\]."
    with pytest.raises(RequiredObjectDataValidationError, match=msg):
        raise RequiredObjectDataValidationError("test", ["data1", "data2"])
