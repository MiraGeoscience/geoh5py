import inspect
from typing import Type

import pytest

from geoh5py import groups
from geoh5py.groups import CustomGroup, Group, GroupType, RootGroup
from geoh5py.objects import ObjectType
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
def test_group_instantiation(group_class: Type[Group]):
    the_workspace = Workspace()

    group_type = group_class.find_or_create_type(the_workspace)
    isinstance(group_type, GroupType)
    assert group_type.workspace is the_workspace
    assert group_type.uid == group_class.default_type_uid()
    assert the_workspace.find_type(group_type.uid, GroupType) is group_type
    assert GroupType.find(the_workspace, group_type.uid) is group_type

    # searching for the wrong type
    assert the_workspace.find_type(group_type.uid, ObjectType) is None

    if the_workspace.root is not None:
        type_used_by_root = the_workspace.root.entity_type is group_type
    created_group = group_class(group_type, name="test group")
    assert created_group.uid is not None
    assert created_group.uid.int != 0
    assert created_group.name == "test group"
    assert created_group.entity_type is group_type

    # should find the type instead of re-creating one
    assert group_class.find_or_create_type(the_workspace) is group_type

    _can_find(the_workspace, created_group)

    # now, make sure that unused data and types do not remain reference in the workspace
    group_type_uid = group_type.uid
    group_type = None  # type: ignore
    # group_type is still referenced by created_group, so it should be tracked by the workspace
    assert the_workspace.find_type(group_type_uid, GroupType) is not None

    created_group_uid = created_group.uid
    created_group = None  # type: ignore
    # no more reference on create_group, so it should be gone from the workspace
    assert the_workspace.find_group(created_group_uid) is None

    if type_used_by_root:
        # type is still used by the workspace root, so still tracked by the workspace
        assert the_workspace.find_type(group_type_uid, GroupType) is not None
    else:
        # no more reference on group_type, so it should be gone from the workspace
        assert the_workspace.find_type(group_type_uid, GroupType) is None


def test_custom_group_instantiation():
    with pytest.raises(RuntimeError):
        assert CustomGroup.default_type_uid() is None

    the_workspace = Workspace()
    with pytest.raises(RuntimeError):
        # cannot get a pre-defined type for a CustomGroup
        CustomGroup.find_or_create_type(the_workspace)

    group_type = GroupType.create_custom(
        the_workspace, name="test custom", description="test custom description"
    )
    assert group_type.name == "test custom"
    assert group_type.description == "test custom description"

    isinstance(group_type, GroupType)
    assert group_type.workspace is the_workspace
    # GroupType.create_custom() uses the generate UUID for the group as its class ID
    assert the_workspace.find_type(group_type.uid, GroupType) is group_type
    assert GroupType.find(the_workspace, group_type.uid) is group_type

    created_group = CustomGroup(group_type, name="test custom group")
    assert created_group.uid is not None
    assert created_group.uid.int != 0
    assert created_group.name == "test custom group"
    assert created_group.entity_type is group_type

    _can_find(the_workspace, created_group)

    # now, make sure that unused data and types do not remain reference in the workspace
    group_type_uid = group_type.uid
    group_type = None
    # group_type is referenced by created_group, so it should survive in the workspace
    assert the_workspace.find_type(group_type_uid, GroupType) is not None

    created_group_uid = created_group.uid
    created_group = None
    # no more reference on group_type, so it should be gone from the workspace
    assert the_workspace.find_data(created_group_uid) is None
    # no more reference on created_group, so it should be gone from the workspace
    assert the_workspace.find_type(group_type_uid, GroupType) is None


def _can_find(workspace, created_group):
    """ Make sure we can find the created group in the workspace.
    """
    all_groups = workspace.all_groups()
    assert len(all_groups) == 2
    iter_all_groups = iter(all_groups)
    assert next(iter_all_groups) in [created_group, workspace.root]
    assert next(iter_all_groups) in [created_group, workspace.root]
    assert workspace.find_group(created_group.uid) is created_group
