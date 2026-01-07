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

from . import shared

class FileIOException(Exception):
    message: str | None = ""

class FileFormatException(Exception):
    message: str | None = ""

@dataclass
class Workspace:
    file_path: str | None = ""
    version: shared.VersionNumber | None = None
    distance_unit: shared.DistanceUnit | None = None
    date_created: shared.DateTime | None = None
    date_modified: shared.DateTime | None = None

class WorkspaceService:
    def get_api_version(
        self,
    ) -> shared.VersionString: ...
    def create_geoh5(
        self,
        file_path: str,
    ) -> Workspace: ...
    def open_geoh5(
        self,
        file_path: str,
    ) -> Workspace: ...
    def save(
        self,
        file_path: str,
        overwrite_file: bool,
    ) -> Workspace: ...
    def export_objects(
        self,
        objects_or_groups: list[shared.Uuid],
        file_path: str,
        overwrite_file: bool,
    ) -> Workspace: ...
    def export_all(
        self,
        file_path: str,
        overwrite_file: bool,
    ) -> Workspace: ...
    def close(
        self,
    ) -> None: ...
    def get_contributors(
        self,
    ) -> list[str]: ...
