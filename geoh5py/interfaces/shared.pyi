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
# pylint: disable=unused-argument,no-self-use,no-name-in-module
# flake8: noqa

from __future__ import annotations

from dataclasses import dataclass

class InvalidUid(Exception):
    message: str | None = ""

class BadEntityType(Exception):
    message: str | None = ""

class BadEntityName(Exception):
    message: str | None = ""

@dataclass
class VersionString:
    value: str | None = ""

@dataclass
class VersionNumber:
    value: float | None = 0.0

@dataclass
class Uuid:
    id: str | None = ""

@dataclass
class DateTime:
    value: str | None = ""

@dataclass
class DistanceUnit:
    unit: str | None = ""

@dataclass
class Coord3D:
    x: float | None = 0.0
    y: float | None = 0.0
    z: float | None = 0.0

@dataclass
class Entity:
    uid: Uuid | None = None
    type_uid: Uuid | None = None
    name: str | None = None
    visible: bool | None = False
    allow_delete: bool | None = True
    allow_rename: bool | None = True
    is_public: bool | None = True

class EntityService:
    def set_public(
        self,
        entities: list[Uuid],
        is_public: bool,
    ) -> None: ...
    def set_visible(
        self,
        entities: list[Uuid],
        visible: bool,
    ) -> None: ...
    def set_allow_delete(
        self,
        entities: list[Uuid],
        allow: bool,
    ) -> None: ...
    def set_allow_rename(
        self,
        entities: list[Uuid],
        allow: bool,
    ) -> None: ...
    def rename(
        self,
        entities: Uuid,
        new_name: str,
    ) -> None: ...
