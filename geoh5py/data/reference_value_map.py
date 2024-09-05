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

    MAP_DTYPE = np.dtype([("Key", "i1"), ("Value", "<U13")])

    def __init__(
        self,
        value_map: dict[int, str] | np.ndarray | tuple,
        name: str = "Value map",
    ):
        self._map: np.ndarray = self.validate_value_map(value_map)
        self.name = name

    def __getitem__(self, item: int) -> str:

        return dict(self._map)[item]

    def __setitem__(self, key, value):
        # verify if it corresponds to boolean values
        if np.all(self.map == BOOLEAN_VALUE_MAP):
            raise AssertionError("Boolean value map cannot be modified")

        if key not in self._map["Key"]:
            raise KeyError(f"Key '{key}' not found in value map.")

        index = list(self._map["Key"]).index(key)

        if key == 0 and value != "Unknown":
            raise ValueError("Value for key 0 must be 'Unknown'")

        self._map[index] = (key, value)

    def __len__(self):
        return len(self._map)

    def __call__(self):
        return self._map

    @classmethod
    def validate_value_map(cls, value_map: np.ndarray | dict) -> np.ndarray:
        """
        Verify that the key and value are valid.
        It raises errors if there is an issue

        :param value_map: Array of key, value pairs.
        """
        if isinstance(value_map, tuple):
            value_map = dict(value_map)

        if isinstance(value_map, np.ndarray) and value_map.dtype.names is None:

            if value_map.ndim == 1:
                value_map = {i: str(val) for i, val in enumerate(set(value_map))}

            value_map = dict(value_map)

        if isinstance(value_map, dict):
            value_map = np.array(list(value_map.items()), dtype=cls.MAP_DTYPE)

        if not isinstance(value_map, np.ndarray):
            raise TypeError("Value map must be a numpy array or dict.")

        if value_map.dtype != cls.MAP_DTYPE:
            raise ValueError(f"Array of 'value_map' must be of dtype = {cls.MAP_DTYPE}")

        if not all(value_map["Key"] >= 0):
            raise KeyError("Key must be an positive integer")

        return value_map

    @property
    def map(self) -> np.ndarray:
        """
        A reference dictionary mapping values to strings.
        The keys are positive integers, and the values description.
        The key '0' is always 'Unknown'.
        """
        return self._map


BOOLEAN_VALUE_MAP = np.array(
    [(0, "False"), (1, "True")],
    dtype=ReferenceValueMap.MAP_DTYPE,
)
