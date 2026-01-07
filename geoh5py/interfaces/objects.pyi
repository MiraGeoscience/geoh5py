# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2020-2026 Mira Geoscience Ltd.                                '
#                                                                              '
#  This file is part of geoh5py.                                               '
#                                                                              '
#  geoh5py is free software: you can redistribute it and/or modify             '
#  it under the terms of the GNU Lesser General Public License as published by '
#  the Free Software Foundation, either version 3 of the License, or           '
#  (at your option) any later version.                                         '
#                                                                              '
#  geoh5py is distributed in the hope that it will be useful,                  '
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              '
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               '
#  GNU Lesser General Public License for more details.                         '
#                                                                              '
#  You should have received a copy of the GNU Lesser General Public License    '
#  along with geoh5py.  If not, see <https://www.gnu.org/licenses/>.           '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# pylint: skip-file
# pylint: disable=unused-argument,no-self-use,no-name-in-module,too-many-public-methods
# flake8: noqa

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from . import shared

class InvalidObjectOperation(Exception):
    message: str | None = ""

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
    entity_: shared.Entity | None = None
    allow_move: bool | None = True

@dataclass
class Points:
    base_: Object | None = None

@dataclass
class Curve:
    base_: Object | None = None

@dataclass
class Surface:
    base_: Object | None = None

@dataclass
class Grid2D:
    base_: Object | None = None

@dataclass
class Drillhole:
    base_: Object | None = None

@dataclass
class BlockModel:
    base_: Object | None = None

@dataclass
class Octree:
    base_: Object | None = None

@dataclass
class GeoImage:
    base_: Object | None = None

@dataclass
class Label:
    base_: Object | None = None

@dataclass
class ObjectQuery:
    name: str | None = ""
    type_id: shared.Uuid | None = None
    in_group: shared.Uuid | None = None
    recursive: bool | None = False

@dataclass
class GeometryTransformation:
    translation: shared.Coord3D | None = None
    rotation_deg: float | None = 0.0

class ObjectsService:
    def get_type(
        self,
        object_class: int,
    ) -> shared.Uuid: ...
    def get_class(
        self,
        type_uid: shared.Uuid,
    ) -> int: ...
    def get_all(
        self,
    ) -> list[Object]: ...
    def find(
        self,
        query: ObjectQuery,
    ) -> list[Object]: ...
    def set_allow_move(
        self,
        objects: list[shared.Uuid],
        allow: bool,
    ) -> None: ...
    def move_to_group(
        self,
        objects: list[shared.Uuid],
        destination_group: shared.Uuid,
    ) -> None: ...
    def get(
        self,
        uid: shared.Uuid,
    ) -> Object: ...
    def narrow_points(
        self,
        uid: shared.Uuid,
    ) -> Points: ...
    def narrow_curve(
        self,
        uid: shared.Uuid,
    ) -> Curve: ...
    def narrow_surface(
        self,
        uid: shared.Uuid,
    ) -> Surface: ...
    def narrow_grid2d(
        self,
        uid: shared.Uuid,
    ) -> Grid2D: ...
    def narrow_drillhole(
        self,
        uid: shared.Uuid,
    ) -> Drillhole: ...
    def narrow_blockmodel(
        self,
        uid: shared.Uuid,
    ) -> BlockModel: ...
    def narrow_octree(
        self,
        uid: shared.Uuid,
    ) -> Octree: ...
    def narrow_geoimage(
        self,
        uid: shared.Uuid,
    ) -> GeoImage: ...
    def narrow_label(
        self,
        uid: shared.Uuid,
    ) -> Label: ...
    def create_any_object(
        self,
        type_id: shared.Uuid,
        name: str,
        parent_group: shared.Uuid,
        attributes: dict[str, str],
    ) -> Object: ...
    def transform(
        self,
        objects: list[shared.Uuid],
        transformation: GeometryTransformation,
    ) -> None: ...
    def set_public(
        self,
        entities: list[shared.Uuid],
        is_public: bool,
    ) -> None: ...
    def set_visible(
        self,
        entities: list[shared.Uuid],
        visible: bool,
    ) -> None: ...
    def set_allow_delete(
        self,
        entities: list[shared.Uuid],
        allow: bool,
    ) -> None: ...
    def set_allow_rename(
        self,
        entities: list[shared.Uuid],
        allow: bool,
    ) -> None: ...
    def rename(
        self,
        entities: shared.Uuid,
        new_name: str,
    ) -> None: ...
