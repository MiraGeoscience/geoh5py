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

from abc import ABC


class ReferenceValueMap(ABC):
    """Maps from reference index to reference value of ReferencedData."""

    def __init__(self, color_map: dict[int, str] | None = None):
        self._map = {} if color_map is None else color_map

    def __getitem__(self, item: int) -> str:
        return self._map[item]

    def __setitem__(self, key, value):
        self._map[key] = value
        return self._map[key]

    def __len__(self):
        return len(self._map)

    def __call__(self):
        return self._map

    @property
    def map(self):
        """
        :obj:`dict`: A reference dictionary mapping values to strings
        """
        return self._map

    @map.setter
    def map(self, value):
        assert isinstance(value, dict), "Map values must be a dictionary"
        self._map = value
