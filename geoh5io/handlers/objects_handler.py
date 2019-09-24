from typing import Dict
from typing import List

from geoh5io import interfaces


# pylint: disable=too-many-public-methods
class ObjectsHandler:
    def get_type(self, object_class: int) -> interfaces.shared.Uuid:
        # TODO
        pass

    def get_class(self, type_uid: interfaces.shared.Uuid) -> int:
        # TODO
        pass

    @staticmethod
    def get_all() -> List[interfaces.objects.Object]:
        # TODO: get from workspace
        # return geoh5io.workspace.Workspace.instance().all_objects()
        return []

    def find(
        self, query: interfaces.objects.ObjectQuery
    ) -> List[interfaces.objects.Object]:
        # TODO
        pass

    def set_allow_move(
        self, objects: List[interfaces.shared.Uuid], allow: bool
    ) -> None:
        # TODO
        pass

    def move_to_group(
        self,
        objects: List[interfaces.shared.Uuid],
        destination_group: interfaces.shared.Uuid,
    ) -> None:
        # TODO
        pass

    def get(self, uid: interfaces.shared.Uuid) -> interfaces.objects.Object:
        # TODO
        pass

    def narrow_points(self, uid: interfaces.shared.Uuid) -> interfaces.objects.Points:
        # TODO
        pass

    def narrow_curve(self, uid: interfaces.shared.Uuid) -> interfaces.objects.Curve:
        # TODO
        pass

    def narrow_surface(self, uid: interfaces.shared.Uuid) -> interfaces.objects.Surface:
        # TODO
        pass

    def narrow_grid2d(self, uid: interfaces.shared.Uuid) -> interfaces.objects.Grid2D:
        # TODO
        pass

    def narrow_drillhole(
        self, uid: interfaces.shared.Uuid
    ) -> interfaces.objects.Drillhole:
        # TODO
        pass

    def narrow_blockmodel(
        self, uid: interfaces.shared.Uuid
    ) -> interfaces.objects.BlockModel:
        # TODO
        pass

    def narrow_octree(self, uid: interfaces.shared.Uuid) -> interfaces.objects.Octree:
        # TODO
        pass

    def narrow_geoimage(
        self, uid: interfaces.shared.Uuid
    ) -> interfaces.objects.GeoImage:
        # TODO
        pass

    def narrow_label(self, uid: interfaces.shared.Uuid) -> interfaces.objects.Label:
        # TODO
        pass

    def create_any_object(
        self,
        type_uid: interfaces.shared.Uuid,
        name: str,
        parent_group: interfaces.shared.Uuid,
        attributes: Dict[str, str],
    ) -> interfaces.objects.Object:
        # TODO
        pass

    def transform(
        self,
        objects: List[interfaces.shared.Uuid],
        transformation: interfaces.objects.GeometryTransformation,
    ) -> None:
        # TODO
        pass

    def set_public(
        self, entities: List[interfaces.shared.Uuid], is_public: bool
    ) -> None:
        # TODO
        pass

    def set_visible(
        self, entities: List[interfaces.shared.Uuid], visible: bool
    ) -> None:
        # TODO
        pass

    def set_allow_delete(
        self, entities: List[interfaces.shared.Uuid], allow: bool
    ) -> None:
        # TODO
        pass

    def set_allow_rename(
        self, entities: List[interfaces.shared.Uuid], allow: bool
    ) -> None:
        # TODO
        pass

    def rename(self, entities: interfaces.shared.Uuid, new_name: str) -> None:
        # TODO
        pass

    @staticmethod
    def test_roundtrip(
        coords: List[interfaces.shared.Coord3D]
    ) -> List[interfaces.shared.Coord3D]:
        print(f"test_roundtrip(len={len(coords)})")
        return [
            interfaces.shared.Coord3D(
                coord.x + 100 if coord.x is not None else None,
                coord.y + 200 if coord.y is not None else None,
                coord.z + 300 if coord.z is not None else None,
            )
            for coord in coords
        ]

    @staticmethod
    def test_roundtrip2(
        coords: interfaces.objects.CoordList3
    ) -> interfaces.objects.CoordList3:
        print(f"test_roundtrip2(len={len(coords.x)})")
        return interfaces.objects.CoordList3(
            [x + 100 for x in coords.x],
            [y + 200 for y in coords.y],
            [z + 300 for z in coords.z],
        )
