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
    TypeValidationError,
    UUIDValidationError,
    ValueValidationError,
)
from geoh5py.ui_json.enforcers import TypeEnforcer, UUIDEnforcer, ValueEnforcer


def test_type_enforcer():
    enforcer = TypeEnforcer(validations=int)
    assert enforcer.validations == [int]
    enforcer.enforce("test", 1)
    msg = "Type 'float' provided for 'test' is invalid. Must be: 'int'."
    with pytest.raises(TypeValidationError, match=msg):
        enforcer.enforce("test", 1.0)

    enforcer = TypeEnforcer(validations=[int, str])
    assert enforcer.validations == [int, str]


def test_value_enforcer():
    enforcer = ValueEnforcer(validations=[1, 2, 3])
    enforcer.enforce("test", 1)
    msg = "Value '4' provided for 'test' is invalid. " "Must be one of: '1', '2', '3'."
    with pytest.raises(ValueValidationError, match=msg):
        enforcer.enforce("test", 4)


def test_uuid_enforcer():
    enforcer = UUIDEnforcer()
    enforcer.enforce("test", str(uuid.uuid4()))
    msg = "Parameter 'test' with value 'notachance' " "is not a valid uuid string."
    with pytest.raises(UUIDValidationError, match=msg):
        enforcer.enforce("test", "notachance")
