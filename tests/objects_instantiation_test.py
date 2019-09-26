import inspect
from typing import Type

import pytest

from geoh5io import objects
from geoh5io.objects import ObjectBase, ObjectType
from geoh5io.workspace import Workspace


def all_object_types():
    for _, obj in inspect.getmembers(objects):
        if inspect.isclass(obj) and issubclass(obj, ObjectBase) and obj is not ObjectBase:
            yield obj


@pytest.mark.parametrize("object_class", all_object_types())
def test_object_instantiation(object_class: Type[ObjectBase]):
    the_workspace = Workspace()

    object_type = object_class.find_or_create_type(the_workspace)
    isinstance(object_type, ObjectType)
    assert object_type.workspace is the_workspace
    assert object_type.uid == object_class.static_type_uid()
    if object_class.static_class_id() is None:
        assert object_type.class_id == object_type.uid
    else:
        assert object_type.class_id == object_class.static_class_id()

    created_object = object_class(object_type, "test")
    assert created_object.uid is not None
    assert created_object.uid.int != 0
    assert created_object.name == "test"
    assert created_object.entity_type is object_type

    # should find the type instead of re-creating one
    object_type2 = object_class.find_or_create_type(the_workspace)
    assert object_type2 is object_type
