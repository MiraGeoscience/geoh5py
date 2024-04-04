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

BOOLEAN_VALUE_MAP = {0: "False", 1: "True"}


class ReferenceValueMap(ABC):
    """Maps from reference index to reference value of ReferencedData."""

    def __init__(self, value_map: dict[int, str]):
        self.map: dict[int, str] = value_map

    def __getitem__(self, item: int) -> str:
        return self._map[item]

    def __setitem__(self, key, value):
        # verify if it corresponds to boolean values
        if self.map == BOOLEAN_VALUE_MAP:
            raise AssertionError("Boolean value map cannot be modified")

        self._validate_key_value(key, value)
        self._map[key] = value

    def __len__(self):
        return len(self._map)

    def __call__(self):
        return self._map

    @staticmethod
    def _validate_key_value(key: int, value: str):
        """
        Verify that the key and value are valid.
        It raises errors if there is an issue

        :param key: The key to verify.
        :param value: The value to verify.
        """
        if not isinstance(key, (int, np.int16, np.integer)) or key < 0:
            raise KeyError("Key must be an positive integer")
        if not isinstance(value, str):
            raise TypeError("Value must be a string")

        if key == 0 and value != "Unknown":
            raise ValueError("Value for key 0 must be 'Unknown'")

    @property
    def map(self) -> dict[int, str]:
        """
        A reference dictionary mapping values to strings.
        The keys are positive integers, and the values description.
        The key '0' is always 'Unknown'.
        """
        return self._map

    @map.setter
    def map(self, value_map: dict[int, str]):
        if not isinstance(value_map, dict):
            raise TypeError("Map values must be a dictionary")
        if value_map != BOOLEAN_VALUE_MAP:
            for key, val in value_map.items():
                self._validate_key_value(key, val)

            if 0 not in value_map:
                value_map[0] = "Unknown"

        self._map = value_map
