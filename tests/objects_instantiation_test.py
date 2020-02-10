import inspect
from typing import Type

import pytest

from geoh5io import objects
from geoh5io.groups import GroupType
from geoh5io.objects import ObjectBase, ObjectType
from geoh5io.workspace import Workspace


def all_object_types():
    for _, obj in inspect.getmembers(objects):
        if (
            inspect.isclass(obj)
            and issubclass(obj, ObjectBase)
            and obj is not ObjectBase
        ):
            yield obj


@pytest.mark.parametrize("object_class", all_object_types())
def test_object_instantiation(object_class: Type[ObjectBase]):
    the_workspace = Workspace()

    object_type = object_class.find_or_create_type(the_workspace)
    isinstance(object_type, ObjectType)
    assert object_type.workspace is the_workspace
    assert object_type.uid == object_class.default_type_uid()
    assert ObjectType.find(the_workspace, object_type.uid) is object_type
    assert the_workspace.find_type(object_type.uid, ObjectType) is object_type

    # searching for the wrong type
    assert the_workspace.find_type(object_type.uid, GroupType) is None

    created_object = object_class(object_type, name="test")
    assert created_object.uid is not None
    assert created_object.uid.int != 0
    assert created_object.name == "test"
    assert created_object.entity_type is object_type

    # should find the type instead of re-creating one
    assert object_class.find_or_create_type(the_workspace) is object_type

    _can_find(the_workspace, created_object)

    # now, make sure that unused data and types do not remain reference in the workspace
    object_type_uid = object_type.uid
    object_type = None  # type: ignore
    # object_type is still referenced by created_group, so it should be tracked by the workspace
    assert the_workspace.find_type(object_type_uid, ObjectType) is not None

    created_object_uid = created_object.uid
    created_object = None  # type: ignore
    # no more reference on created_object, so it should be gone from the workspace
    assert the_workspace.find_object(created_object_uid) is None

    # no more reference on object_type, so it should be gone from the workspace
    assert the_workspace.find_type(object_type_uid, ObjectType) is None


def _can_find(workspace, created_object):
    """ Make sure we can find the created object in the workspace.
    """
    all_objects = workspace.all_objects()
    assert len(all_objects) == 1
    assert next(iter(all_objects)) is created_object
    assert workspace.find_object(created_object.uid) is created_object
