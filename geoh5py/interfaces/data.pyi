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
from enum import IntEnum

from . import shared

class InvalidDataOperation(Exception):
    message: str | None = ""

class BadPrimitiveType(Exception):
    message: str | None = ""

class DataAssociation(IntEnum):
    UNKNOWN = 0
    OBJECT = 1
    CELL = 2
    FACE = 3
    VERTEX = 4

class PrimitiveType(IntEnum):
    UNKNOWN = 0
    INTEGER = 1
    FLOAT = 2
    REFERENCED = 3
    TEXT = 4
    FILENAME = 5
    DATETIME = 6
    BLOB = 7

@dataclass
class Data:
    entity_: shared.Entity | None = None
    association: int | None = None

@dataclass
class DataUnit:
    unit: str | None = ""

@dataclass
class DataType:
    uid: shared.Uuid | None = None
    name: str | None = None
    description: str | None = ""
    units: DataUnit | None = None
    primitive_type: int | None = None

@dataclass
class DataSlab:
    start: int | None = 0
    stride: int | None = 1
    count: int | None = 0
    block: int | None = 1

@dataclass
class ReferencedDataEntry:
    key: int | None = None
    value: str | None = None

@dataclass
class ReferencedValues:
    indices: list[int] | None = None
    entries: list[ReferencedDataEntry] | None = None

@dataclass
class DataQuery:
    name: str | None = None
    object_or_group: shared.Uuid | None = None
    data_type: shared.Uuid | None = None
    primitive_type: int | None = None
    association: int | None = None

@dataclass
class DataTypeQuery:
    name: str | None = None
    primitive_type: int | None = None
    units: DataUnit | None = None

class DataService:
    def get_all(
        self,
    ) -> list[Data]: ...
    def find(
        self,
        query: DataQuery,
    ) -> list[Data]: ...
    def get(
        self,
        uid: shared.Uuid,
    ) -> Data: ...
    def get_float_values(
        self,
        data: shared.Uuid,
        slab: DataSlab,
    ) -> list[float]: ...
    def get_integer_values(
        self,
        data: shared.Uuid,
        slab: DataSlab,
    ) -> list[int]: ...
    def get_text_values(
        self,
        data: shared.Uuid,
        slab: DataSlab,
    ) -> list[str]: ...
    def get_referenced_values(
        self,
        data: shared.Uuid,
        slab: DataSlab,
    ) -> ReferencedValues: ...
    def get_datetime_values(
        self,
        data: shared.Uuid,
        slab: DataSlab,
    ) -> list[str]: ...
    def get_filename_values(
        self,
        data: shared.Uuid,
        slab: DataSlab,
    ) -> list[str]: ...
    def get_file_content(
        self,
        data: shared.Uuid,
        file_name: str,
    ) -> str: ...
    def get_blob_values(
        self,
        data: shared.Uuid,
        slab: DataSlab,
    ) -> list[int]: ...
    def get_blob_element(
        self,
        data: shared.Uuid,
        index: int,
    ) -> str: ...
    def get_all_types(
        self,
    ) -> list[DataType]: ...
    def find_types(
        self,
        query: DataTypeQuery,
    ) -> list[DataType]: ...
    def get_type(
        self,
        uid: shared.Uuid,
    ) -> DataType: ...
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
