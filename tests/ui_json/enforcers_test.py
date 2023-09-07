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
    UUIDValidationError,
    ValueValidationError,
)
from geoh5py.ui_json.enforcers import (
    EnforcerPool,
    TypeEnforcer,
    UUIDEnforcer,
    ValueEnforcer,
)


def test_enforcer_pool_construction():
    pool = EnforcerPool("my_param")
    assert pool.name == "my_param"
    assert pool.enforcers == []
    pool = EnforcerPool("my_param", [TypeEnforcer(str)])
    assert pool.enforcers == [TypeEnforcer(str)]


def test_enforcer_pool_update():
    pool = EnforcerPool("my_param")
    pool.update({"type": str, "value": "onlythis"})
    assert pool.enforcers == [TypeEnforcer(str), ValueEnforcer("onlythis")]
    pool.update({"type": int, "value": "nowonlythis"}, protected=["type"])
    assert pool.enforcers == [TypeEnforcer(str), ValueEnforcer("nowonlythis")]


def test_enforcer_pool_validations():
    validations = {"type": [str], "value": "onlythis"}
    pool = EnforcerPool.from_validations("my_param", validations)
    assert pool.validations == validations


def test_enforcer_pool_from_validations():
    pool = EnforcerPool.from_validations("my_param", {"type": str})
    assert pool.enforcers == [TypeEnforcer(str)]


def test_enforcer_pool_raises_single_error():
    enforcers = EnforcerPool("my_param", [TypeEnforcer(str)])
    enforcers.validate("1")
    msg = "Type 'int' provided for 'my_param' is invalid. "
    msg += "Must be: 'str'."
    with pytest.raises(TypeValidationError, match=msg):
        enforcers.validate(1)


def test_enforcer_pool_raises_aggregate_error():
    enforcers = EnforcerPool(
        "my_param", [TypeEnforcer(str), ValueEnforcer(["onlythis"])]
    )
    enforcers.validate("onlythis")
    msg = (
        "Validation of 'my_param' collected 2 errors:\n\t"
        "0. Type 'int' provided for 'my_param' is invalid"
    )
    with pytest.raises(AggregateValidationError, match=msg):
        enforcers.validate(1)


def test_enforcer_str():
    enforcer = TypeEnforcer(validations=str)
    assert str(enforcer) == "<TypeEnforcer> : [<class 'str'>]"


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
    msg = "Value '4' provided for 'test' is invalid. "
    msg += "Must be one of: '1', '2', '3'."
    with pytest.raises(ValueValidationError, match=msg):
        enforcer.enforce("test", 4)


def test_uuid_enforcer():
    enforcer = UUIDEnforcer()
    enforcer.enforce("test", str(uuid.uuid4()))
    msg = "Parameter 'test' with value 'notachance' "
    msg += "is not a valid uuid string."
    with pytest.raises(UUIDValidationError, match=msg):
        enforcer.enforce("test", "notachance")
