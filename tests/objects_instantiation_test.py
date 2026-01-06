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

import inspect

import pytest

from geoh5py import objects
from geoh5py.groups import GroupType
from geoh5py.objects import CellObject, DrapeModel, ObjectBase, ObjectType
from geoh5py.workspace import Workspace


def all_object_types():
    for _, obj in inspect.getmembers(objects):
        if (
            inspect.isclass(obj)
            and issubclass(obj, ObjectBase)
            and obj not in (ObjectBase, CellObject)
        ):
            yield obj


@pytest.mark.parametrize("object_class", all_object_types())
def test_object_instantiation(object_class):
    if object_class.default_type_uid() is None:
        # this object class is not instantiable
        return

    with Workspace() as workspace:
        object_type = object_class.find_or_create_type(workspace)

        if isinstance(object_class, type(DrapeModel)):
            return

        isinstance(object_type, ObjectType)
        assert object_type.workspace is workspace
        assert ObjectType.find(workspace, object_type.uid) is object_type
        assert workspace.find_type(object_type.uid, ObjectType) is object_type

        # searching for the wrong type
        assert workspace.find_type(object_type.uid, GroupType) is None

        created_object = object_class(object_type, name="test")
        assert created_object.uid is not None
        assert created_object.uid.int != 0
        assert created_object.name == "test"
        assert created_object.entity_type is object_type

        # should find the type instead of re-creating one
        assert object_class.find_or_create_type(workspace) is object_type

        _can_find(workspace, created_object)

        # now, make sure that unused data and types do not remain reference in the workspace
        object_type_uid = object_type.uid
        object_type = None  # type: ignore
        # object_type is still referenced by created_group, so it should be tracked by the workspace
        assert workspace.find_type(object_type_uid, ObjectType) is not None

        created_object_uid = created_object.uid
        created_object = None  # type: ignore
        # no more reference on created_object, so it should be gone from the workspace
        assert workspace.find_object(created_object_uid) is None

        # no more reference on object_type, so it should be gone from the workspace
        assert workspace.find_type(object_type_uid, ObjectType) is None


def _can_find(workspace, created_object):
    """Make sure we can find the created object in the workspace."""
    all_objects = workspace.objects
    assert len(all_objects) == 1
    assert next(iter(all_objects)) is created_object
    assert workspace.find_object(created_object.uid) is created_object
