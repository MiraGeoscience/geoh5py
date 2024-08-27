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
        value_map: dict[int, str] | np.ndarray,
        data_maps: dict[str, np.ndarray] | None = None,
    ):
        self._map: np.ndarray = self.validate_value_map(value_map)
        self.data_maps: dict[str, dict] | None = data_maps

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
        if isinstance(value_map, dict):
            value_map = np.array(list(value_map.items()), dtype=cls.MAP_DTYPE)

        if not isinstance(value_map, np.ndarray):
            raise TypeError("Value map must be a numpy array or dict.")

        if value_map.dtype != cls.MAP_DTYPE:
            raise ValueError(f"Array of 'value_map' must be of dtype = {cls.MAP_DTYPE}")

        if not all(value_map["Key"] >= 0):
            raise KeyError("Key must be an positive integer")

        if set(value_map["Value"]) == {"False", "True"}:
            return value_map

        if 0 not in value_map["Key"]:
            value_map.resize(len(value_map) + 1, refcheck=False)
            value_map[-1] = (0, "Unknown")

        if dict(value_map)[0] != "Unknown":
            raise ValueError("Value for key 0 must be 'Unknown'")

        return value_map

    @property
    def map(self) -> np.ndarray:
        """
        A reference dictionary mapping values to strings.
        The keys are positive integers, and the values description.
        The key '0' is always 'Unknown'.
        """
        return self._map

    @property
    def data_maps(self) -> dict[str, np.ndarray] | None:
        """
        A reference dictionary mapping properties to numpy arrays.
        """
        return self._data_maps

    @data_maps.setter
    def data_maps(self, value: dict[str, np.ndarray] | None):
        if value is not None:
            if not isinstance(value, dict):
                raise TypeError("Property maps must be a dictionary")
            for key, val in value.items():
                if not isinstance(val, np.ndarray):
                    raise TypeError(
                        f"Property maps values for '{key}' must be a numpy array."
                    )

        self._data_maps = value


BOOLEAN_VALUE_MAP = np.array(
    [(0, "False"), (1, "True")],
    dtype=ReferenceValueMap.MAP_DTYPE,
)
