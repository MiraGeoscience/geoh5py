from typing import Dict
from typing import List

from geoh5io import interfaces


class ObjectsHandler:
    def get_type(self, object_class: int) -> interfaces.shared.Uuid:
        # TODO
        pass

    def get_class(self, type: interfaces.shared.Uuid) -> int:
        # TODO
        pass

    def get_all(self,) -> List[interfaces.objects.Object]:
        # TODO
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

    def narrowPoints(self, uid: interfaces.shared.Uuid) -> interfaces.objects.Points:
        # TODO
        pass

    def narrowCurve(self, uid: interfaces.shared.Uuid) -> interfaces.objects.Curve:
        # TODO
        pass

    def narrowSurface(self, uid: interfaces.shared.Uuid) -> interfaces.objects.Surface:
        # TODO
        pass

    def narrowGrid2D(self, uid: interfaces.shared.Uuid) -> interfaces.objects.Grid2D:
        # TODO
        pass

    def narrowDrillhole(
        self, uid: interfaces.shared.Uuid
    ) -> interfaces.objects.Drillhole:
        # TODO
        pass

    def narrowBlockModel(
        self, uid: interfaces.shared.Uuid
    ) -> interfaces.objects.BlockModel:
        # TODO
        pass

    def narrowOctree(self, uid: interfaces.shared.Uuid) -> interfaces.objects.Octree:
        # TODO
        pass

    def narrowGeoImage(
        self, uid: interfaces.shared.Uuid
    ) -> interfaces.objects.GeoImage:
        # TODO
        pass

    def narrowLabel(self, uid: interfaces.shared.Uuid) -> interfaces.objects.Label:
        # TODO
        pass

    def createAnyObject(
        self,
        type: interfaces.shared.Uuid,
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
