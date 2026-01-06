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

import re
from pathlib import Path

import numpy as np
import pytest

from geoh5py.objects import Points
from geoh5py.shared.exceptions import (
    AssociationValidationError,
    AtLeastOneValidationError,
    OptionalValidationError,
    PropertyGroupValidationError,
    RequiredValidationError,
    ShapeValidationError,
    TypeValidationError,
    UUIDValidationError,
    ValueValidationError,
)
from geoh5py.shared.validators import (
    AssociationValidator,
    AtLeastOneValidator,
    PropertyGroupValidator,
    RequiredValidator,
    ShapeValidator,
    TypeValidator,
    UUIDValidator,
    ValueValidator,
)
from geoh5py.ui_json import InputFile
from geoh5py.workspace import Workspace


def test_validation_types():
    validation_types = [
        "association",
        "property_group_type",
        "one_of",
        "required",
        "shape",
        "types",
        "uuid",
        "values",
    ]

    errs = [
        AssociationValidator(),
        PropertyGroupValidator(),
        AtLeastOneValidator(),
        RequiredValidator(),
        ShapeValidator(),
        TypeValidator(),
        UUIDValidator(),
        ValueValidator(),
    ]

    for i, err in enumerate(errs):
        assert err.validator_type == validation_types[i]


def test_association_validator(tmp_path: Path):
    workspace = Workspace.create(tmp_path / r"test.geoh5")
    workspace2 = Workspace.create(tmp_path / r"test2.geoh5")
    points = Points.create(workspace, vertices=np.array([[1, 2, 3], [4, 5, 6]]))
    points2 = Points.create(workspace2, vertices=np.array([[1, 2, 3], [4, 5, 6]]))
    validator = AssociationValidator()

    # Test valid workspace
    with pytest.raises(AssociationValidationError) as excinfo:
        validator("test", points, workspace2)
    assert AssociationValidationError.message("test", points, workspace2) == str(
        excinfo.value
    )

    # Test valid points object
    with pytest.raises(AssociationValidationError) as excinfo:
        validator("test", points, points2)
    assert AssociationValidationError.message("test", points, points2) == str(
        excinfo.value
    )

    # Test for wrong valid type
    with pytest.raises(ValueError) as excinfo:
        validator("test", points, "nogood")
    assert "Provided 'nogood' " in str(excinfo.value)

    # No validation error for none value or valid
    validator("test", None, points)
    validator("test", points, None)


def test_property_group_validator(tmp_path):
    workspace = Workspace.create(tmp_path / r"test.geoh5")
    points = Points.create(
        workspace, vertices=np.array([[1, 2, 3], [4, 5, 6]]), name="test_points"
    )
    test_data = points.add_data({"points_data": {"values": np.array([1.0, 2.0])}})
    property_group = points.add_data_to_group(test_data, "test_group")
    validator = PropertyGroupValidator()

    with pytest.raises(PropertyGroupValidationError) as excinfo:
        validator("test", property_group, "not_test_group")
    assert PropertyGroupValidationError.message(
        "test", property_group, ["not_test_group"]
    ) == str(excinfo.value)


def test_required_validator():
    validator = RequiredValidator()
    with pytest.raises(RequiredValidationError) as excinfo:
        validator("test", None, True)
    assert RequiredValidationError.message("test", None, None) == str(excinfo.value)


def test_shape_validator():
    validator = ShapeValidator()
    with pytest.raises(ShapeValidationError) as excinfo:
        validator("test", [[1, 2, 3], [4, 5, 6]], (3, 2))
    assert ShapeValidationError.message("test", (2,), (3, 2)) == str(excinfo.value)

    # No validation error for None
    validator("test", None, (3, 2))


def test_type_validator():
    validator = TypeValidator()

    with pytest.raises(
        TypeError,
        match=re.escape("Input `valid` options must be a type or list of types."),
    ):
        validator("test", 3, 123)

    # Test non-iterable value, single valid
    with pytest.raises(
        TypeValidationError,
        match=TypeValidationError.message("test", int.__name__, [type({}).__name__]),
    ):
        validator("test", 3, type({}))

    # Test non-iterable value, more than one valid
    with pytest.raises(
        TypeValidationError,
        match=TypeValidationError.message(
            "test", int.__name__, [str.__name__, type({}).__name__]
        ),
    ):
        validator("test", 3, [str, type({})])

    # Test iterable value single valid both invalid
    with pytest.raises(
        TypeValidationError,
        match=TypeValidationError.message("test", int.__name__, [type({}).__name__]),
    ):
        validator("test", [3, 2], [type({})])

    # Test iterable value single valid one valid, one invalid
    with pytest.raises(
        TypeValidationError,
        match=TypeValidationError.message("test", str.__name__, [int.__name__]),
    ):
        validator("test", [3, "a"], [int])


def test_uuid_validator():
    validator = UUIDValidator()

    # Test bad uid string
    with pytest.raises(UUIDValidationError) as excinfo:
        validator("test", "sdr")
    assert UUIDValidationError.message("test", "sdr", None) == str(excinfo.value)

    # No validation error for None
    validator("test", None, [])


def test_value_validator():
    validator = ValueValidator()
    with pytest.raises(ValueValidationError) as excinfo:
        validator("test", "blah", ["nope", "not here"])
    assert ValueValidationError.message("test", "blah", ["nope", "not here"]) == str(
        excinfo.value
    )

    # No validation error for None
    validator("test", None, ["these", "don't", "matter"])


def test_validate_data(tmp_path):
    with Workspace.create(str(tmp_path / r"test.geoh5")):
        pass

    ui_json = {
        "title": "test",
        "geoh5": str(tmp_path / r"test.geoh5"),
        "param_1": {
            "label": "param_1",
            "optional": True,
            "enabled": False,
            "value": None,
        },
        "param_2": {
            "label": "param_2",
            "optional": True,
            "enabled": False,
            "value": None,
        },
    }
    validations = {
        "param_1": {"one_of": "sad little parameter", "types": [str, type(None)]},
        "param_2": {"one_of": "sad little parameter", "types": [str, type(None)]},
    }

    with pytest.raises(
        AtLeastOneValidationError, match="at least one sad little parameter"
    ):
        InputFile(ui_json=ui_json, validations=validations).data

    ui_json["param_1"].update({"enabled": True})

    with pytest.raises(OptionalValidationError, match="Cannot set a None"):
        InputFile(ui_json=ui_json, validations=validations).data
