import inspect
from typing import Type

import pytest

from geoh5io import data
from geoh5io.data import Data, DataAssociationEnum, DataType
from geoh5io.workspace import Workspace


def all_data_types():
    for _, obj in inspect.getmembers(data):
        if inspect.isclass(obj) and issubclass(obj, Data) and obj is not Data:
            yield obj


@pytest.mark.parametrize("data_class", all_data_types())
def test_data_instantiation(data_class: Type[Data]):
    the_workspace = Workspace()

    data_type = DataType.create(the_workspace, data_class)
    assert data_type.uid is not None
    assert data_type.uid.int != 0
    assert data_type.name is None
    assert data_type.units is None
    assert data_type.primitive_type == data_class.primitive_type()

    assert the_workspace.find_type(data_type.uid, DataType) is data_type

    created_data = data_class(data_type, DataAssociationEnum.VERTEX, "test")
    assert created_data.uid is not None
    assert created_data.uid.int != 0
    assert created_data.name == "test"
    assert created_data.association == DataAssociationEnum.VERTEX

    all_data = the_workspace.all_data()
    assert len(all_data) == 1
    assert next(iter(all_data)) is created_data
    assert the_workspace.find_data(created_data.uid) is created_data
