# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2025-2026 Mira Geoscience Ltd.                                '
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
from enum import IntEnum

from . import shared

class InvalidGroupOperation(Exception):
    message: str | None = ""

class GroupClass(IntEnum):
    UNKNOWN = 0
    CONTAINER = 1
    DRILLHOLE = 2

@dataclass
class Group:
    entity_: shared.Entity | None = None
    allow_move: bool | None = True

@dataclass
class GroupQuery:
    name: str | None = None
    type_uid: shared.Uuid | None = None
    in_group: shared.Uuid | None = None
    recursive: bool | None = False

class GroupsService:
    def get_root(
        self,
    ) -> Group: ...
    def get_type(
        self,
        group_class: int,
    ) -> shared.Uuid: ...
    def get_class(
        self,
        type_uid: shared.Uuid,
    ) -> int: ...
    def get_all(
        self,
    ) -> list[Group]: ...
    def find(
        self,
        query: GroupQuery,
    ) -> list[Group]: ...
    def set_allow_move(
        self,
        groups: list[shared.Uuid],
        allow: bool,
    ) -> None: ...
    def move_to_group(
        self,
        groups: list[shared.Uuid],
        destination_group: shared.Uuid,
    ) -> None: ...
    def create(
        self,
        type_uid: shared.Uuid,
    ) -> Group: ...
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
