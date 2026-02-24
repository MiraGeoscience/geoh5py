# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2020-2026 Mira Geoscience Ltd.                                     '
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
import uuid

import pytest

from geoh5py import groups
from geoh5py.groups import CustomGroup, Group, GroupType, RootGroup
from geoh5py.objects import ObjectType
from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace


def all_group_types():
    for _, obj in inspect.getmembers(groups):
        if (
            inspect.isclass(obj)
            and issubclass(obj, Group)
            and obj not in [Group, CustomGroup, RootGroup]
        ):
            yield obj


@pytest.mark.parametrize("group_class", all_group_types())
def test_group_instantiation(group_class, tmp_path):
    h5file_path = tmp_path / f"{__name__}.geoh5"
    with Workspace.create(h5file_path) as workspace:
        group_type = group_class.find_or_create_type(workspace)
        isinstance(group_type, GroupType)
        assert group_type.workspace is workspace
        assert group_type.uid == group_class.default_type_uid()
        assert workspace.find_type(group_type.uid, GroupType) is group_type
        assert GroupType.find(workspace, group_type.uid) is group_type

        # searching for the wrong type
        assert workspace.find_type(group_type.uid, ObjectType) is None

        type_used_by_root = False
        if workspace.root is not None:
            type_used_by_root = workspace.root.entity_type is group_type
        created_group = group_class(entity_type=group_type, name="test group")
        assert created_group.uid is not None
        assert created_group.uid.int != 0
        assert created_group.name == "test group"
        assert created_group.entity_type is group_type

        # should find the type instead of re-creating one
        assert group_class.find_or_create_type(workspace) is group_type

        _can_find(workspace, created_group)

        # now, make sure that unused data and types do not remain reference in the workspace
        group_type_uid = group_type.uid
        group_type = None  # type: ignore
        # group_type is still referenced by created_group, so it should be tracked by the workspace
        assert workspace.find_type(group_type_uid, GroupType) is not None

        created_group_uid = created_group.uid
        workspace.remove_entity(created_group)
        created_group = None  # type: ignore
        # no more reference on create_group, so it should be gone from the workspace
        assert workspace.find_group(created_group_uid) is None

        if type_used_by_root:
            # type is still used by the workspace root, so still tracked by the workspace
            assert workspace.find_type(group_type_uid, GroupType) is not None
        else:
            # no more reference on group_type, so it should be gone from the workspace
            assert workspace.find_type(group_type_uid, GroupType) is None


def test_custom_group_instantiation(tmp_path):
    assert isinstance(CustomGroup.default_type_uid(), uuid.UUID)

    h5file_path = tmp_path / f"{__name__}.geoh5"
    with Workspace.create(h5file_path) as workspace:
        group_type = GroupType.create_custom(
            workspace, name="test custom", description="test custom description"
        )
        assert group_type.name == "test custom"
        assert group_type.description == "test custom description"

        isinstance(group_type, GroupType)
        assert group_type.workspace is workspace
        # GroupType.create_custom() uses the generate UUID for the group as its class ID
        assert workspace.find_type(group_type.uid, GroupType) is group_type
        assert GroupType.find(workspace, group_type.uid) is group_type

        created_group = CustomGroup(
            entity_type=group_type, name="test custom group", parent=workspace.root
        )
        workspace.save_entity(created_group)
        assert created_group.uid is not None
        assert created_group.uid.int != 0
        assert created_group.name == "test custom group"
        assert created_group.entity_type is group_type

        _can_find(workspace, created_group)

        # now, make sure that unused data and types remains referenced in the workspace
        group_type_uid = group_type.uid
        group_type = None
        assert workspace.find_type(group_type_uid, GroupType) is not None

    with Workspace(h5file_path) as new_workspace:
        rec_group = new_workspace.get_entity("test custom group")[0]

    compare_entities(
        rec_group,
        created_group,
        ignore=["_parent", "_metadata"],
    )


def _can_find(workspace, created_group):
    """Make sure we can find the created group in the workspace."""
    all_groups = workspace.groups
    assert len(all_groups) == 2
    iter_all_groups = iter(all_groups)
    assert next(iter_all_groups) in [created_group, workspace.root]
    assert next(iter_all_groups) in [created_group, workspace.root]
    assert workspace.find_group(created_group.uid) is created_group
