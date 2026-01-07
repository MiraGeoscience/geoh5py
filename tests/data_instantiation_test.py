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

import inspect

import numpy as np
import pytest

from geoh5py import data
from geoh5py.data import (
    Data,
    DataAssociationEnum,
    NumericData,
)
from geoh5py.data.data_type import DataType
from geoh5py.objects import ObjectType, Points
from geoh5py.shared import FLOAT_NDV, INTEGER_NDV
from geoh5py.workspace import Workspace


COLOUR_NO_DATA = np.array(
    [
        (90, 90, 90, 0),
    ],
    dtype=[("r", "u1"), ("g", "u1"), ("b", "u1"), ("a", "u1")],
)


def all_data_types():
    for _, obj in inspect.getmembers(data):
        if (
            inspect.isclass(obj)
            and issubclass(obj, Data)
            and obj not in (Data, NumericData)
        ):
            yield obj


@pytest.mark.parametrize("data_class", all_data_types())
def test_data_instantiation(data_class, tmp_path):
    h5file_path = tmp_path / f"{__name__}.geoh5"
    with Workspace.create(h5file_path) as workspace:
        data_type = DataType(workspace, primitive_type=data_class)
        assert data_type.uid is not None
        assert data_type.uid.int != 0
        assert data_type.name == "Entity"
        assert data_type.units is None
        assert issubclass(data_class, data_type.primitive_type.value)
        assert workspace.find_type(data_type.uid, DataType) is data_type
        assert DataType.find(workspace, data_type.uid) is data_type

        # searching for the wrong type
        assert workspace.find_type(data_type.uid, ObjectType) is None

        created_data = data_class(
            entity_type=data_type, association=DataAssociationEnum.VERTEX, name="test"
        )

        assert created_data.uid is not None
        assert created_data.uid.int != 0
        assert created_data.name == "test"
        assert created_data.association == DataAssociationEnum.VERTEX

        _can_find(workspace, created_data)

        if isinstance(created_data.nan_value, np.ndarray):
            assert all(created_data.nan_value == COLOUR_NO_DATA)
        else:
            assert created_data.nan_value in [
                None,
                0,
                np.nan,
                INTEGER_NDV,
                FLOAT_NDV,
                "",
            ]

        # now, make sure that unused data and types do not remain reference in the workspace
        data_type_uid = data_type.uid
        data_type = None  # type: ignore
        # data_type is still referenced by created_data, so it should survive in the workspace
        assert workspace.find_type(data_type_uid, DataType) is not None

        created_data_uid = created_data.uid

        if created_data.allow_delete:
            workspace.remove_entity(created_data)
            created_data = None  # type: ignore
            # no more reference on created_data, so it should be gone from the workspace
            assert workspace.find_data(created_data_uid) is None

            # no more reference on data_type, so it should be gone from the workspace
            assert workspace.find_type(data_type_uid, DataType) is None

        with pytest.raises(TypeError, match="Input 'entity_type' with primitive_type"):
            data.TextData(data_type="bidon")

        with pytest.raises(NotImplementedError, match="Only add_data"):
            workspace.validate_data_type({}, object())


def _can_find(workspace, created_data):
    """Make sure we can find the created data in the workspace."""
    all_data = workspace.data
    assert len(all_data) == 1
    assert next(iter(all_data)) is created_data
    assert workspace.find_data(created_data.uid) is created_data


def test_scale_continuity(tmp_path):
    h5file_path = tmp_path / r"testPoints.geoh5"
    with Workspace.create(h5file_path) as workspace:
        points = Points.create(
            workspace,
            name="point",
            vertices=np.vstack((np.arange(20), np.arange(20), np.zeros(20))).T,
            allow_move=False,
        )
        data = points.add_data(
            {"DataValues": {"association": "VERTEX", "values": np.random.randn(20)}}
        )
        data.entity_type.scale = "Log"
        data.entity_type.precision = 4.0
        data.entity_type.scientific_notation = True

    with Workspace(h5file_path) as workspace:
        points = workspace.get_entity("point")[0]
        data = points.get_data("DataValues")[0]

        assert data.entity_type.scale == "Log"
        assert data.entity_type.precision == 4
        assert data.entity_type.scientific_notation is True


def test_copy_from_extent(tmp_path):
    # Generate a random cloud of points
    h5file_path = tmp_path / r"testPoints.geoh5"
    workspace = Workspace.create(h5file_path)
    points = Points.create(
        workspace,
        vertices=np.vstack((np.arange(20), np.arange(20), np.zeros(20))).T,
        allow_move=False,
    )

    data_ = points.add_data(
        {"DataValues": {"association": "VERTEX", "values": np.random.randn(20)}}
    )

    cropped_data = data_.copy_from_extent(np.vstack([[5, 5], [10, 10]]))

    # prepare validation
    validation = data_.values
    validation[:5] = np.nan
    validation[11:] = np.nan

    np.testing.assert_array_equal(cropped_data.values, validation)


def test_data_type_attributes():
    workspace = Workspace()
    data_type = DataType(workspace, primitive_type="REFERENCED")

    with pytest.raises(ValueError, match=r"Attribute 'mapping' should be one"):
        data_type.mapping = "exponential"

    with pytest.raises(TypeError, match=r"Attribute 'hidden' must be a bool"):
        data_type.hidden = "abc"

    with pytest.raises(
        TypeError, match=r"Attribute 'duplicate_type_on copy' must be a bool"
    ):
        data_type.duplicate_type_on_copy = "abc"

    with pytest.raises(
        ValueError, match=r"Attribute 'number_of_bins' should be an integer"
    ):
        data_type.number_of_bins = "abc"

    with pytest.raises(ValueError, match="greater than 0 or None"):
        data_type.number_of_bins = 0

    with pytest.raises(
        ValueError, match=r"Attribute 'primitive_type' value must be of type"
    ):
        data_type.primitive_type = "FLOAT"

    with pytest.raises(
        TypeError, match="Attribute 'transparent_no_data' must be a bool"
    ):
        data_type.transparent_no_data = "FLOAT"

    with pytest.raises(TypeError, match=r"Attribute 'units' must be a string"):
        data_type.units = 1

    with pytest.raises(ValueError, match=r"Attribute 'primitive_type' should be one"):
        data_type.validate_primitive_type(1)

    with pytest.raises(TypeError, match=r"Attribute 'duplicate_on_copy'"):
        data_type.duplicate_on_copy = "bidon"

    with pytest.raises(TypeError, match=r"Attribute 'precision'"):
        data_type.precision = "bidon"

    with pytest.raises(ValueError, match=r"Attribute 'scale'"):
        data_type.scale = "bidon"

    with pytest.raises(TypeError, match=r"Attribute 'scientific_notation'"):
        data_type.scientific_notation = "bidon"


def test_add_data_increment_names(tmp_path):
    """
    Test that adding data with incrementing names works correctly.
    """
    h5file_path = tmp_path / r"testAddDataIncrementNames.geoh5"
    with Workspace.create(h5file_path) as workspace:
        points = Points.create(
            workspace,
            vertices=np.vstack((np.arange(20), np.arange(20), np.zeros(20))).T,
            allow_move=False,
        )

        # Add data with the same name multiple times
        for _ in range(3):
            points.add_data(
                {"DataValues": {"association": "VERTEX", "values": np.random.randn(20)}}
            )

        # Check that the names are incremented correctly
        assert points.get_data_list() == [
            "DataValues",
            "DataValues(1)",
            "DataValues(2)",
        ]
