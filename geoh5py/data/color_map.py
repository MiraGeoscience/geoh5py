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

from typing import TYPE_CHECKING

import numpy as np

from ..shared.exceptions import ShapeValidationError
from ..shared.utils import map_attributes


if TYPE_CHECKING:
    from ..workspace import Workspace
    from .data_type import DataType


class ColorMap:
    """Records colors assigned to value ranges (where Value is the start of the range)."""

    _attribute_map = {"File name": "name"}
    _names = ["Value", "Red", "Green", "Blue", "Alpha"]
    _formats = ["<f8", "u1", "u1", "u1", "u1"]

    def __init__(self, **kwargs):
        self.parent = None
        self.name = "geoh5py_custom.TBL"
        self.values = np.empty((0, 5))

        map_attributes(self, **kwargs)

    @property
    def values(self) -> np.ndarray:
        """
        :obj:`numpy.array`: Colormap defined by values and corresponding RGBA:

        .. code-block:: python

            values = [
                [V_1, R_1, G_1, B_1, A_1],
                ..., [V_i, R_i, G_i, B_i, A_i]
            ]

        where V (Values) are sorted floats defining the position of each RGBA.
        R (Red), G (Green), B (Blue) and A (Alpha) are integer values between [0, 255].
        """
        return np.vstack([self._values[name] for name in self._names])

    @values.setter
    def values(self, values: np.ndarray):
        if not isinstance(values, np.ndarray):
            raise TypeError(f"Input 'values' of ColorMap must be of type {np.ndarray}.")

        if np.issubdtype(values.dtype.base, np.number):
            if values.shape[1] != 5:
                raise ShapeValidationError("values", values.shape, "(*, 5)")

            self._values = np.core.records.fromarrays(
                values.T, names=self._names, formats=self._formats
            )

        else:
            if values.dtype.names is None or not all(
                name in self._names for name in values.dtype.names
            ):
                raise ValueError(
                    f"Input 'values' must contain fields with types {self._names}"
                )

            self._values = np.asarray(
                values, dtype=list(zip(self._names, self._formats, strict=False))
            )

        if self.workspace is not None and self.parent is not None:
            self.workspace.update_attribute(self.parent, "color_map")  # pylint: disable=no-member

    @property
    def name(self) -> str:
        """
        :obj:`str`: Name of the colormap
        """
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = str(value)
        if self.parent is not None:
            self.parent.workspace.update_attribute(self.parent, "color_map")

    @property
    def parent(self) -> DataType | None:
        """Parent data type"""
        return self._parent

    @parent.setter
    def parent(self, data_type: DataType | None):
        self._parent = data_type

    @property
    def workspace(self) -> Workspace | None:
        """Workspace object"""
        if self.parent is not None:
            return self.parent.workspace

        return None

    def __len__(self):
        return len(self._values)
