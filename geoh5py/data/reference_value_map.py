#  Copyright (c) 2024 Mira Geoscience Ltd.
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

import numpy as np


class ReferenceValueMap(ABC):
    """Maps from reference index to reference value of ReferencedData."""

    def __init__(self, color_map: dict[int, str]):
        self.map: dict[int, str] = color_map

    def __getitem__(self, item: int) -> str:
        return self._map[item]

    def __setitem__(self, key, value):
        if not isinstance(key, (int, np.int32)) or key < 0:
            raise KeyError("Key must be an positive integer")
        if key == 0:
            if value != "Unknown" and not (
                len(self) <= 2 and value == "False" and value == "True"
            ):
                raise ValueError("Value for key 0 must be 'Unknown'")
        if not isinstance(value, str):
            raise TypeError("Value must be a string")

        self._map[key] = value

    def __len__(self):
        return len(self._map)

    def __call__(self):
        return self._map

    @property
    def map(self) -> dict[int, str]:
        """
        A reference dictionary mapping values to strings.
        The keys are positive integers, and the values description.
        The key '0' is always 'Unknown'.
        """
        return self._map

    @map.setter
    def map(self, value: dict[int, str]):
        if not isinstance(value, dict):
            raise TypeError("Map values must be a dictionary")

        if not all(isinstance(k, (int, np.int32)) and k >= 0 for k in value.keys()):
            raise KeyError("Map keys must be positive integers")

        if 0 in value.keys():
            if value[0] != "Unknown" and not (
                len(value) <= 2 and value[0] == "False" and value[1] == "True"
            ):
                raise ValueError("Map value for 0 must be 'Unknown'")
        else:
            value[0] = "Unknown"

        self._map = value
