#  Copyright (c) 2021 Mira Geoscience Ltd.
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

import inspect
import tempfile
from pathlib import Path

import pytest

from geoh5py import data
from geoh5py.data import Data, DataAssociationEnum, DataType
from geoh5py.objects import ObjectType
from geoh5py.workspace import Workspace


def all_data_types():
    for _, obj in inspect.getmembers(data):
        if inspect.isclass(obj) and issubclass(obj, Data) and obj is not Data:
            yield obj


@pytest.mark.parametrize("data_class", all_data_types())
def test_data_instantiation(data_class):
    # TODO: no file on disk should be required for this test
    #       as workspace does not have to be saved
    with tempfile.TemporaryDirectory() as tempdir:
        the_workspace = Workspace(Path(tempdir) / f"{__name__}.geoh5")

        data_type = DataType.create(the_workspace, data_class)
        assert data_type.uid is not None
        assert data_type.uid.int != 0
        assert data_type.name == str(data_type.uid)
        assert data_type.units is None
        assert data_type.primitive_type == data_class.primitive_type()
        assert the_workspace.find_type(data_type.uid, DataType) is data_type
        assert DataType.find(the_workspace, data_type.uid) is data_type

        # searching for the wrong type
        assert the_workspace.find_type(data_type.uid, ObjectType) is None

        created_data = data_class(
            data_type, association=DataAssociationEnum.VERTEX, name="test"
        )
        assert created_data.uid is not None
        assert created_data.uid.int != 0
        assert created_data.name == "test"
        assert created_data.association == DataAssociationEnum.VERTEX

        _can_find(the_workspace, created_data)

        # now, make sure that unused data and types do not remain reference in the workspace
        data_type_uid = data_type.uid
        data_type = None  # type: ignore
        # data_type is still referenced by created_data, so it should survive in the workspace
        assert the_workspace.find_type(data_type_uid, DataType) is not None

        created_data_uid = created_data.uid
        created_data = None  # type: ignore
        # no more reference on created_data, so it should be gone from the workspace
        assert the_workspace.find_data(created_data_uid) is None

        # no more reference on data_type, so it should be gone from the workspace
        assert the_workspace.find_type(data_type_uid, DataType) is None


def _can_find(workspace, created_data):
    """Make sure we can find the created data in the workspace."""
    all_data = workspace.all_data()
    assert len(all_data) == 1
    assert next(iter(all_data)) is created_data
    assert workspace.find_data(created_data.uid) is created_data
