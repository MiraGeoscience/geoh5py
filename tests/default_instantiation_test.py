from geoh5io.data import (
    DataAssociationEnum,
    DataType,
    PrimitiveTypeEnum,
    ReferencedData,
)
from geoh5io.groups import ContainerGroup
from geoh5io.objects import Points
from geoh5io.workspace import Workspace, active_workspace


def test_point_instantiation():
    with active_workspace(Workspace()):
        pts = Points("test")
        assert pts.uid is not None
        assert pts.name == "test"


def test_group_instantiation():
    with active_workspace(Workspace()):
        grp = ContainerGroup("test")
        assert grp.uid is not None
        assert grp.name == "test"


def test_data_instantiation():
    with active_workspace(Workspace()):
        data_type = DataType.create(ReferencedData)
        assert data_type.uid is not None
        assert data_type.name is None
        assert data_type.units is None
        assert data_type.primitive_type == PrimitiveTypeEnum.REFERENCED

        data = ReferencedData(data_type, DataAssociationEnum.VERTEX, "test")
        assert data.primitive_type() == DataAssociationEnum.VERTEX
        assert data.uid is not None
        assert data.name == "test"
        assert data.association == DataAssociationEnum.VERTEX
