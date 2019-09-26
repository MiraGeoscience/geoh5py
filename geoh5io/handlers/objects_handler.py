from typing import TYPE_CHECKING, Dict, List

from geoh5io import interfaces

if TYPE_CHECKING:
    from geoh5io.interfaces.objects import Object as i_Object
    from geoh5io.interfaces.objects import ObjectQuery as i_ObjectQuery
    from geoh5io.interfaces.objects import Points as i_Points
    from geoh5io.interfaces.objects import Curve as i_Curve
    from geoh5io.interfaces.objects import Surface as i_Surface
    from geoh5io.interfaces.objects import Grid2D as i_Grid2D
    from geoh5io.interfaces.objects import BlockModel as i_BlockModel
    from geoh5io.interfaces.objects import Drillhole as i_Drillhole
    from geoh5io.interfaces.objects import GeoImage as i_GeoImage
    from geoh5io.interfaces.objects import Octree as i_Octree
    from geoh5io.interfaces.objects import Label as i_Label
    from geoh5io.interfaces.objects import (
        GeometryTransformation as i_GeometryTransformation,
    )
    from geoh5io.interfaces.shared import Uuid as i_Uuid

# pylint: disable=too-many-public-methods
class ObjectsHandler:
    def get_type(self, object_class: int) -> i_Uuid:
        # TODO
        pass

    def get_class(self, type_uid: i_Uuid) -> int:
        # TODO
        pass

    @staticmethod
    def get_all() -> List[i_Object]:
        # TODO: get from workspace
        # return geoh5io.workspace.Workspace.instance().all_objects()
        return []

    def find(self, query: i_ObjectQuery) -> List[i_Object]:
        # TODO
        pass

    def set_allow_move(self, objects: List[i_Uuid], allow: bool) -> None:
        # TODO
        pass

    def move_to_group(self, objects: List[i_Uuid], destination_group: i_Uuid) -> None:
        # TODO
        pass

    def get(self, uid: i_Uuid) -> i_Object:
        # TODO
        pass

    def narrow_points(self, uid: i_Uuid) -> i_Points:
        # TODO
        pass

    def narrow_curve(self, uid: i_Uuid) -> i_Curve:
        # TODO
        pass

    def narrow_surface(self, uid: i_Uuid) -> i_Surface:
        # TODO
        pass

    def narrow_grid2d(self, uid: i_Uuid) -> i_Grid2D:
        # TODO
        pass

    def narrow_drillhole(self, uid: i_Uuid) -> i_Drillhole:
        # TODO
        pass

    def narrow_blockmodel(self, uid: i_Uuid) -> i_BlockModel:
        # TODO
        pass

    def narrow_octree(self, uid: i_Uuid) -> i_Octree:
        # TODO
        pass

    def narrow_geoimage(self, uid: i_Uuid) -> i_GeoImage:
        # TODO
        pass

    def narrow_label(self, uid: i_Uuid) -> i_Label:
        # TODO
        pass

    def create_any_object(
        self,
        type_uid: i_Uuid,
        name: str,
        parent_group: i_Uuid,
        attributes: Dict[str, str],
    ) -> i_Object:
        # TODO
        pass

    def transform(
        self, objects: List[i_Uuid], transformation: i_GeometryTransformation
    ) -> None:
        # TODO
        pass

    def set_public(self, entities: List[i_Uuid], is_public: bool) -> None:
        # TODO
        pass

    def set_visible(self, entities: List[i_Uuid], visible: bool) -> None:
        # TODO
        pass

    def set_allow_delete(self, entities: List[i_Uuid], allow: bool) -> None:
        # TODO
        pass

    def set_allow_rename(self, entities: List[i_Uuid], allow: bool) -> None:
        # TODO
        pass

    def rename(self, entities: i_Uuid, new_name: str) -> None:
        # TODO
        pass
