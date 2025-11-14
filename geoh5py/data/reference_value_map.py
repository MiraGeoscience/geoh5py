# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2025 Mira Geoscience Ltd.                                     '
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


from __future__ import annotations

import numpy as np
from h5py import special_dtype

from geoh5py.shared.utils import find_unique_name


class ReferenceValueMap:
    """Maps from reference index to reference value of ReferencedData."""

    MAP_DTYPE = np.dtype([("Key", "<u4"), ("Value", special_dtype(vlen=str))])

    def __init__(
        self,
        value_map: dict[int, str] | np.ndarray | tuple,
        name: str = "Value map",
    ):
        self._map: np.ndarray = self.validate_value_map(value_map)
        self.name = name

    def __getitem__(self, item: int) -> str:
        return dict(self._map)[item]

    def __len__(self) -> int:
        return len(self._map)

    def __call__(self) -> dict:
        try:
            map_string = self._map.astype(np.dtype([("Key", "<u4"), ("Value", "U25")]))
        except UnicodeDecodeError:
            map_string = self._map

        return dict(map_string)

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
                value_map = cls._dict_map_from_value_array(value_map)

            value_map = dict(value_map)

        if isinstance(value_map, dict):
            if not np.all(np.asarray(list(value_map)) >= 0):
                raise KeyError("Key must be an positive integer")

            # Make sure no duplicated name as case-insensitive
            names = list(value_map.values())
            ids = list(value_map)
            value_list = []
            for ind, name in enumerate(names):
                new = find_unique_name(name, names[:ind], case_sensitive=False)
                value_list.append((ids[ind], new))

            value_map = np.array(
                value_list, dtype=[("Key", "<u4"), ("Value", special_dtype(vlen=str))]
            )

            # TODO: Replace with numpy.dtypes.StringDType instead for support of variable
            # length string after moving to numpy >=2.0
            str_len = [
                len(val) if isinstance(val, str) else len(str(val))
                for val in value_map["Value"]
            ]

            value_map["Value"] = np.char.encode(
                value_map["Value"].astype(f"U{max(str_len) if str_len else 32}"),
                "utf-8",
            )

        if not isinstance(value_map, np.ndarray):
            raise TypeError("Value map must be a numpy array or dict.")

        if value_map.dtype != cls.MAP_DTYPE:
            raise ValueError(f"Array of 'value_map' must be of dtype = {cls.MAP_DTYPE}")

        return value_map

    @property
    def map(self) -> np.ndarray:
        """
        A reference array mapping values to strings.
        The keys are positive integers, and the values description.
        The key '0' is always 'Unknown'.
        """
        return self._map

    def map_values(self, values: np.ndarray) -> np.ndarray:
        """
        Map the values to the reference values.

        :param values: The values to map.

        :return: The mapped values.
        """
        mapper = np.sort(self.map, order="Key")
        indices = np.searchsorted(mapper["Key"], values)
        return mapper["Value"][indices]

    @staticmethod
    def _dict_map_from_value_array(value_array: np.ndarray) -> dict:
        """
        Create a map from an array of values.

        :param value_array: The array of values to map.

        :return: A dictionary mapping indices to values.
        """
        unique_set = set(value_array)

        if np.issubdtype(value_array.dtype, np.number):
            unique_set.discard(0)
            return {int(val): f"Unit {val}" for val in unique_set}

        return {i + 1: str(value) for i, value in enumerate(value_array)}


BOOLEAN_VALUE_MAP = np.array(
    [(0, b"False"), (1, b"True")],
    dtype=ReferenceValueMap.MAP_DTYPE,
)
