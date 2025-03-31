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

from ..shared.utils import min_max_scaler
from .data import Data
from .data_association_enum import DataAssociationEnum
from .primitive_type_enum import PrimitiveTypeEnum


logger = logging.getLogger(__name__)


class Colour(Data):
    """
    Base class for the ternary colour map.

    Warning: actually, GA is storing RGBA but ignore the "A" band (full 255).
    To mimic this behavior, works with RGB only (and drop A if exists).
    """

    allowed_parent = [
        "Points",
        "Curve",
        "BlockModel",
        "Octree",
        "Grid2D",
    ]

    def __init__(self, *args, **kwargs):
        self._verify_parents(kwargs.get("parent", None))

        super().__init__(*args, **kwargs)

    @staticmethod
    def _array_to_rgb(values: np.ndarray) -> np.ndarray:
        """
        Convert a 2D numpy array to RGB.

        :param values: 2D numpy array with 3 or 4 bands.

        :return: A structured array with "r", "g", "b" bands
        """
        if values.ndim != 2 or values.shape[1] not in (3, 4):
            raise ValueError("Values must be a 2D numpy array containing RGB bands.")

        if values.dtype != np.uint8:
            values = min_max_scaler(values, 0, 255, axis=0)

        values = (
            values[:, :3]
            .astype(np.uint8)
            .view([("r", "u1"), ("g", "u1"), ("b", "u1")])
            .reshape(-1)
        )

        return values

    @staticmethod
    def _structured_to_rgb(values: np.ndarray) -> np.ndarray:
        """
        Convert a structured array to RGB.

        :param values: The structured array with "r", "g", "b" bands.

        :return: The RGB values.
        """

        for band in ["r", "g", "b"]:
            if band not in values.dtype.names:
                raise ValueError(
                    "Values must be a 2D numpy array containing RGB bands."
                )

            # ensure the min and the max are between 0 and 255
            if not values[band].dtype == np.uint8:
                values[band] = min_max_scaler(values[band], 0, 255).astype(np.uint8)

        return values[["r", "g", "b"]]

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
            if parent_classes not in cls.allowed_parent:
                raise TypeError(
                    f"Parent '{parent_classes}' is not allowed for Colour data.\n"
                    f"The allowed parents are {cls.allowed_parent}"
                )

    def format_length(self, values: np.ndarray) -> np.ndarray:
        """
        Ensure the structured RGB array has the expected length.

        :param values: The structured array to check.

        :return: Structured array with adjusted length.
        """
        if self.n_values is None:
            return values

        dtype = values.dtype
        nan_rgb = tuple([self.nan_value] * len(dtype.names))

        if len(values) < self.n_values:
            full_vector = np.empty(self.n_values, dtype=dtype)
            full_vector[:] = nan_rgb
            full_vector[: len(values)] = values
            return full_vector

        if (
            len(values) > self.n_values
            and self.association is not DataAssociationEnum.OBJECT
        ):
            logger.warning(
                "Input 'values' of shape (%s,) expected. Array of shape %s provided for data %s.",
                self.n_values,
                values.shape,
                self.name,
            )
            return values[: self.n_values]

        return values

    @property
    def nan_value(self):
        return 0

    @classmethod
    def primitive_type(cls) -> PrimitiveTypeEnum:
        return PrimitiveTypeEnum.COLOUR

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
