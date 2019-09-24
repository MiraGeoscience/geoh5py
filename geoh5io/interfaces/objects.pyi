from dataclasses import dataclass
from enum import IntEnum
from typing import *


from . import shared




class InvalidObjectOperation(Exception):
    message: Optional[str] = ""




class ObjectClass(IntEnum):
    UNKNOWN = 0
    POINTS = 1
    CURVE = 2
    SURFACE = 3
    GRID2D = 4
    DRILLHOLE = 5
    BLOCKMODEL = 6
    OCTREE = 7
    GEOIMAGE = 8
    LABEL = 9




@dataclass
class Object:
    entity_: shared.Entity
    allow_move: Optional[bool] = True

@dataclass
class Points:
    base_: Object

@dataclass
class Curve:
    base_: Object

@dataclass
class Surface:
    base_: Object

@dataclass
class Grid2D:
    base_: Object

@dataclass
class Drillhole:
    base_: Object

@dataclass
class BlockModel:
    base_: Object

@dataclass
class Octree:
    base_: Object

@dataclass
class GeoImage:
    base_: Object

@dataclass
class Label:
    base_: Object

@dataclass
class ObjectQuery:
    name: Optional[str] = ""
    type_id: Optional[shared.Uuid] = None
    in_group: Optional[shared.Uuid] = None
    recursive: Optional[bool] = False

@dataclass
class CoordList3:
    x: List[float]
    y: List[float]
    z: List[float]

@dataclass
class GeometryTransformation:
    translation: Optional[shared.Coord3D] = None
    rotation_deg: Optional[float] = 0.0




class ObjectsService:
    def get_type(
        self,
        object_class: int,
    ) -> shared.Uuid:
        ...
    def get_class(
        self,
        type_uid: shared.Uuid,
    ) -> int:
        ...
    def get_all(
        self,
    ) -> List[Object]:
        ...
    def find(
        self,
        query: ObjectQuery,
    ) -> List[Object]:
        ...
    def set_allow_move(
        self,
        objects: List[shared.Uuid],
        allow: bool,
    ) -> None:
        ...
    def move_to_group(
        self,
        objects: List[shared.Uuid],
        destination_group: shared.Uuid,
    ) -> None:
        ...
    def get(
        self,
        uid: shared.Uuid,
    ) -> Object:
        ...
    def narrow_points(
        self,
        uid: shared.Uuid,
    ) -> Points:
        ...
    def narrow_curve(
        self,
        uid: shared.Uuid,
    ) -> Curve:
        ...
    def narrow_surface(
        self,
        uid: shared.Uuid,
    ) -> Surface:
        ...
    def narrow_grid2d(
        self,
        uid: shared.Uuid,
    ) -> Grid2D:
        ...
    def narrow_drillhole(
        self,
        uid: shared.Uuid,
    ) -> Drillhole:
        ...
    def narrow_blockmodel(
        self,
        uid: shared.Uuid,
    ) -> BlockModel:
        ...
    def narrow_octree(
        self,
        uid: shared.Uuid,
    ) -> Octree:
        ...
    def narrow_geoimage(
        self,
        uid: shared.Uuid,
    ) -> GeoImage:
        ...
    def narrow_label(
        self,
        uid: shared.Uuid,
    ) -> Label:
        ...
    def create_any_object(
        self,
        type_id: shared.Uuid,
        name: str,
        parent_group: shared.Uuid,
        attributes: Dict[str, str],
    ) -> Object:
        ...
    def transform(
        self,
        objects: List[shared.Uuid],
        transformation: GeometryTransformation,
    ) -> None:
        ...
    def test_roundtrip(
        self,
        coords: List[shared.Coord3D],
    ) -> List[shared.Coord3D]:
        ...
    def test_roundtrip2(
        self,
        coords: CoordList3,
    ) -> CoordList3:
        ...
    def set_public(
        self,
        entities: List[shared.Uuid],
        is_public: bool,
    ) -> None:
        ...
    def set_visible(
        self,
        entities: List[shared.Uuid],
        visible: bool,
    ) -> None:
        ...
    def set_allow_delete(
        self,
        entities: List[shared.Uuid],
        allow: bool,
    ) -> None:
        ...
    def set_allow_rename(
        self,
        entities: List[shared.Uuid],
        allow: bool,
    ) -> None:
        ...
    def rename(
        self,
        entities: shared.Uuid,
        new_name: str,
    ) -> None:
        ...
