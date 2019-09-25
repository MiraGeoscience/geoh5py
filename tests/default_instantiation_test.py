from geoh5io.data import (
    DataAssociationEnum,
    DataType,
    PrimitiveTypeEnum,
    ReferencedData,
)
from geoh5io.groups import ContainerGroup
from geoh5io.objects import Points
from geoh5io.workspace import Workspace


class TestDefaultInstantiation:
    ws = None

    def setup_method(self):
        # create a new empty workspace not tied to any file
        self._ws = Workspace()

    def teardown_method(self):
        self._ws = None

    def test_point_instantiation(self):
        pts = Points("test")
        assert pts.uid is not None
        assert pts.name == "test"

    def test_group_instantiation(self):
        grp = ContainerGroup("test")
        assert grp.uid is not None
        assert grp.name == "test"

    def test_data_instantiation(self):
        dt = DataType.create(ReferencedData)
        assert dt.uid is not None
        assert dt.name is None
        assert dt.units is None
        assert dt.primitive_type == PrimitiveTypeEnum.REFERENCED

        data = ReferencedData(dt, DataAssociationEnum.VERTEX, "test")
        assert data.primitive_type() == DataAssociationEnum.VERTEX
        assert data.uid is not None
        assert data.name == "test"
        assert data.association == DataAssociationEnum.VERTEX
