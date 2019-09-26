import inspect
from typing import Type

import pytest

from geoh5io import groups
from geoh5io.groups import CustomGroup, Group, GroupType
from geoh5io.workspace import Workspace


def all_group_types():
    for name, obj in inspect.getmembers(groups):
        if (
            inspect.isclass(obj)
            and issubclass(obj, Group)
            and obj not in [Group, CustomGroup]
        ):
            yield obj


@pytest.mark.parametrize("group_class", all_group_types())
def test_group_instantiation(group_class: Type[Group]):
    the_workspace = Workspace()

    group_type = group_class.find_or_create_type(the_workspace)
    isinstance(group_type, GroupType)
    assert group_type.workspace is the_workspace
    assert group_type.uid == group_class.static_type_uid()
    if group_class.static_class_id() is None:
        assert group_type.class_id == group_type.uid
    else:
        assert group_type.class_id == group_class.static_class_id()

    created_group = group_class(group_type, "test group")
    assert created_group.uid is not None
    assert created_group.uid.int != 0
    assert created_group.name == "test group"
    assert created_group.entity_type is group_type

    # should find the type instead of re-creating one
    group_type2 = group_class.find_or_create_type(the_workspace)
    assert group_type2 is group_type


def test_custom_group_instantiation():
    assert CustomGroup.static_type_uid() is None
    assert CustomGroup.static_class_id() is None

    the_workspace = Workspace()
    with pytest.raises(RuntimeError) as error:
        # cannot get a pre-defined type for a CustomGroup
        CustomGroup.find_or_create_type(the_workspace)

    group_type = GroupType.create_custom(the_workspace, "test custom", "test custom description")
    assert group_type.name == "test custom"
    assert group_type.description == "test custom description"

    isinstance(group_type, GroupType)
    assert group_type.workspace is the_workspace
    assert group_type.class_id == group_type.uid

    created_group = CustomGroup(group_type, "test custom group")
    assert created_group.uid is not None
    assert created_group.uid.int != 0
    assert created_group.name == "test custom group"
    assert created_group.entity_type is group_type

    # should be able find the group type again
    assert group_type is GroupType.find(the_workspace, group_type.uid)
