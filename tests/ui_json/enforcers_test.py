# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2020-2026 Mira Geoscience Ltd.                                '
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

import numpy as np
import pytest

from geoh5py import Workspace
from geoh5py.objects import Points
from geoh5py.shared.exceptions import (
    AggregateValidationError,
    RequiredFormMemberValidationError,
    RequiredObjectDataValidationError,
    RequiredUIJsonParameterValidationError,
    RequiredWorkspaceObjectValidationError,
    TypeValidationError,
    UUIDValidationError,
    ValueValidationError,
)
from geoh5py.shared.utils import SetDict
from geoh5py.ui_json.enforcers import (
    EnforcerPool,
    RequiredEnforcer,
    RequiredFormMemberEnforcer,
    RequiredObjectDataEnforcer,
    RequiredUIJsonParameterEnforcer,
    RequiredWorkspaceObjectEnforcer,
    TypeEnforcer,
    UUIDEnforcer,
    ValueEnforcer,
)


def test_enforcer_pool_recruit():
    validations = SetDict(
        type={str},
        value={"onlythis"},
        uuid={None},
        required={"me"},
        required_uijson_parameters={"me", "you"},
        required_form_members={"label", "value"},
        required_workspace_object={"data"},
        required_object_data={"object"},
    )
    enforcers = EnforcerPool._recruit(validations)  # pylint: disable=protected-access

    assert enforcers == [
        TypeEnforcer({str}),
        ValueEnforcer({"onlythis"}),
        UUIDEnforcer({None}),
        RequiredEnforcer({"me"}),
        RequiredUIJsonParameterEnforcer({"me", "you"}),
        RequiredFormMemberEnforcer({"label", "value"}),
        RequiredWorkspaceObjectEnforcer({"data"}),
        RequiredObjectDataEnforcer({"object"}),
    ]


def test_enforcer_pool_construction():
    pool = EnforcerPool("my_param", [TypeEnforcer({str})])
    assert pool.enforcers == [TypeEnforcer({str})]


def test_enforcer_pool_validations():
    validations = SetDict(type=str, value="onlythis")
    pool = EnforcerPool.from_validations("my_param", validations)
    assert pool.validations == validations
    pool = EnforcerPool("my_param", [TypeEnforcer({str}), ValueEnforcer({"onlythis"})])
    assert pool.validations == validations


def test_enforcer_pool_from_validations():
    pool = EnforcerPool.from_validations("my_param", {"type": str})
    assert pool.enforcers == [TypeEnforcer(str)]


def test_enforcer_pool_raises_single_error():
    enforcers = EnforcerPool("my_param", [TypeEnforcer({str})])
    enforcers.enforce("1")
    msg = "Type 'int' provided for 'my_param' is invalid. "
    msg += "Must be: 'str'."
    with pytest.raises(TypeValidationError, match=msg):
        enforcers.enforce(1)


def test_enforcer_pool_raises_aggregate_error():
    enforcers = EnforcerPool(
        "my_param", [TypeEnforcer({str}), ValueEnforcer({"onlythis"})]
    )
    enforcers.enforce("onlythis")
    msg = (
        "Validation of 'my_param' collected 2 errors:\n\t"
        "0. Type 'int' provided for 'my_param' is invalid"
    )
    with pytest.raises(AggregateValidationError, match=msg):
        enforcers.enforce(1)


def test_enforcer_str():
    enforcer = TypeEnforcer(validations={str})
    assert str(enforcer) == "<TypeEnforcer> : {<class 'str'>}"


def test_type_enforcer():
    enforcer = TypeEnforcer(validations={int})
    assert enforcer.validations == {int}
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


def test_required_uijson_parameter_enforcer():
    enforcer = RequiredUIJsonParameterEnforcer(["my_param"])
    msg = r"UIJson: 'my_param' is missing required parameter\(s\): \['my_param'\]."
    with pytest.raises(RequiredUIJsonParameterValidationError, match=msg):
        enforcer.enforce("my_param", {"label": "my param"})


def test_required_form_member_enforcer():
    enforcer = RequiredFormMemberEnforcer(["my_member"])
    msg = r"Form: 'my_member' is missing required member\(s\): \['my_member'\]."
    with pytest.raises(RequiredFormMemberValidationError, match=msg):
        enforcer.enforce("my_member", {"label": "my member"})


def test_required_workspace_object_enforcer(tmp_path):
    geoh5 = Workspace(tmp_path / "working_file.geoh5")
    pts = Points.create(geoh5, vertices=np.random.rand(10, 3), name="my_points")
    other_geoh5 = Workspace(tmp_path / "other_file.geoh5")
    other_pts = Points.create(
        other_geoh5, vertices=np.random.rand(10, 3), name="my_other_points"
    )

    data = {"geoh5": geoh5, "my_points": {"value": pts}}
    validations = ["my_points"]
    enforcer = RequiredWorkspaceObjectEnforcer(validations)
    enforcer.enforce(str(geoh5.h5file.stem), data)

    data["my_points"] = {"value": other_pts}
    msg = r"Workspace: 'working_file' is missing required object\(s\): \['my_points'\]."
    with pytest.raises(RequiredWorkspaceObjectValidationError, match=msg):
        enforcer.enforce(str(geoh5.h5file.stem), data)


def test_required_object_data_enforcer(tmp_path):
    geoh5 = Workspace(tmp_path / "working_file.geoh5")
    pts = Points.create(geoh5, vertices=np.random.rand(10, 3), name="my_points")
    my_data = pts.add_data({"my_data": {"values": np.random.rand(10)}})
    other_pts = Points.create(
        geoh5, vertices=np.random.rand(10, 3), name="my_other_points"
    )
    the_wrong_data = other_pts.add_data(
        {"my_other_data": {"values": np.random.rand(10)}}
    )

    data = {
        "geoh5": geoh5,
        "object": {"value": pts},
        "data": {"value": my_data},
    }
    validations = [("object", "data")]
    enforcer = RequiredObjectDataEnforcer(validations)
    enforcer.enforce(str(geoh5.h5file.stem), data)

    data["data"]["value"] = the_wrong_data
    msg = (
        r"Workspace: 'working_file' object\(s\) \['object'\] "
        r"are missing required children \['data'\]."
    )
    with pytest.raises(RequiredObjectDataValidationError, match=msg):
        enforcer.enforce(str(geoh5.h5file.stem), data)
