#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoh5py.
#
#  geoh5py is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  geoh5py is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with geoh5py.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

from typing import Any

from geoh5py.ui_json.parameters import Parameter


class MemberKeys:
    """Converts in and out of camel (ui.json) and snake (python) case"""

    camel_to_snake: dict[str, str] = {
        "groupOptional": "group_optional",
        "dependencyType": "dependency_type",
        "groupDependency": "group_dependency",
        "groupDependencyType": "group_dependency_type",
        "lineEdit": "line_edit",
        "choiceList": "choice_list",
        "fileDescription": "file_description",
        "fileType": "file_type",
        "fileMulti": "file_multi",
        "meshType": "mesh_type",
        "dataType": "data_type",
        "dataGroupType": "data_group_type",
        "isValue": "is_value",
    }

    @property
    def snake_to_camel(self) -> dict[str, str]:
        return {v: k for k, v in self.camel_to_snake.items()}

    def _map_single(self, key: str, convention: str = "snake"):
        """Map a string from snake to camel or vice versa."""

        if convention == "snake":
            out = self.camel_to_snake.get(key, key)
        elif convention == "camel":
            out = self.snake_to_camel.get(key, key)
        else:
            raise ValueError("Convention must be 'snake' or 'camel'.")

        return out

    def map(self, collection: dict[str, Any], convention="snake"):
        """Map a dictionary from snake to camel or vice versa."""
        return {self._map_single(k, convention): v for k, v in collection.items()}


class MemberParameter(Parameter):
    pass
