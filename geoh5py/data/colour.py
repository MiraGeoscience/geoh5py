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

import logging
from typing import Any

import numpy as np
from numpy.lib import recfunctions as rfn

from ..shared.utils import min_max_scaler
from .data import Data


logger = logging.getLogger(__name__)


class Colour(Data):
    """
    Base class for the ternary colour map.

    Warning: actually, GA is storing RGBA but ignore the "A" band (full 255).
    To mimic this behavior, works with RGB only (and drop A if exists).
    """

    _allowed_parent = [
        "Points",
        "Curve",
        "BlockModel",
        "Octree",
        "Grid2D",
    ]

    _formats = [("r", "u1"), ("g", "u1"), ("b", "u1"), ("a", "u1")]

    _nan_value = np.array([(90, 90, 90, 0)], dtype=_formats)

    def __init__(self, *args, **kwargs):
        self._verify_parents(kwargs.get("parent", None))

        super().__init__(*args, **kwargs)

    @classmethod
    def _array_to_rgb(cls, values: np.ndarray) -> np.ndarray:
        """
        Convert a 2D numpy array to RGB.

        :param values: 2D numpy array with 3 or 4 bands.

        :return: A structured array with "r", "g", "b" bands
        """
        if values.ndim != 2 or values.shape[1] not in (3, 4):
            raise ValueError("Values must be a 2D numpy array containing RGB bands.")

        nan_mask = np.isnan(values).any(axis=1)

        if values.dtype != np.uint8:
            values = min_max_scaler(values, 0, 255, axis=0)

        if values.shape[1] == 3:
            values = np.column_stack(
                (values, np.full(values.shape[0], 255, dtype=np.uint8))
            )

        values = values.astype(np.uint8).view(cls._formats).reshape(-1)

        values[nan_mask] = cls._nan_value

        return values

    @classmethod
    def _structured_to_rgb(cls, values: np.ndarray) -> np.ndarray:
        """
        Convert a structured array to RGB.

        :param values: The structured array with "r", "g", "b" bands.

        :return: The RGB values.
        """

        # convert the dtype names to lower case
        values = values.astype(
            [(name.lower(), values.dtype[name]) for name in values.dtype.names]
        )

        nan_mask = np.zeros(values.shape[0], dtype=bool)

        for band in ["r", "g", "b", "a"]:
            if band not in values.dtype.names:
                if band == "a":
                    values = rfn.append_fields(
                        values,
                        names="a",
                        data=np.full(values.shape[0], 255, dtype=np.uint8),
                        usemask=False,
                    )
                else:
                    raise ValueError(
                        "Values must be a 2D numpy array containing RGB bands."
                    )

            # ensure the min and the max are between 0 and 255
            if values[band].dtype != np.uint8:
                nan_mask = np.logical_or(nan_mask, np.isnan(values[band]))
                values[band] = min_max_scaler(values[band], 0, 255).astype(np.uint8)

        values[nan_mask] = cls._nan_value

        return values[["r", "g", "b", "a"]]

    @classmethod
    def _verify_parents(cls, parent: Any):
        """
        Verify if the parent is allowed for Colour data.

        For now the objects that can use Colour data are limited.
        Raises an error if the parent is not allowed.

        :param parent: The parent object class.
        """
        if parent is not None:
            # use name to avoid circular import
            parent_classes = parent.__class__.__name__
            if parent_classes not in cls._allowed_parent:
                raise TypeError(
                    f"Parent '{parent_classes}' is not allowed for Colour data.\n"
                    f"The allowed parents are {cls._allowed_parent}"
                )

    @property
    def nan_value(self):
        return self._nan_value

    def validate_values(self, values: np.ndarray | None) -> Any:
        """
        Validate values for UnknownData.

        Important Note: GA is only supporting RGB but is storing as RGBA.
        Transform values to RGB always.
        """
        if values is None:
            return values

        if not isinstance(values, np.ndarray):
            raise TypeError(f"Values must be a numpy array. Get {type(values)}")

        # convert to structure array
        if values.dtype.names is None:
            values = self._array_to_rgb(values)
        else:
            values = self._structured_to_rgb(values)

        return self.format_length(values)
