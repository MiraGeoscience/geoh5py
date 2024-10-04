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

from abc import ABC, abstractmethod
from copy import deepcopy
from warnings import warn

import numpy as np

from .data import Data, PrimitiveTypeEnum
from .data_association_enum import DataAssociationEnum


class NumericData(Data, ABC):
    """
    Data container for floats values
    """

    @classmethod
    def primitive_type(cls) -> PrimitiveTypeEnum:
        return PrimitiveTypeEnum.INVALID

    @property
    @abstractmethod
    def ndv(self):
        """No-data-value"""

    @property
    def ndv_values(self) -> np.ndarray | None:
        """
        Data with nan replaced by ndv
        """
        if self.values is None:
            return None
        values = deepcopy(self.values)
        values[np.isnan(values)] = self.ndv

        return values

    @abstractmethod
    def format_type(self, values: np.ndarray) -> np.ndarray:
        """
        Check if the type of values is valid and convert it to right dtype.
        :param values: numpy array to modify.
        :return: the formatted values.
        """

    def format_length(self, values: np.ndarray) -> np.ndarray:
        """
        Check for possible mismatch between the length of values
        :param values: the values to check.
        :return: the values with the right length.
        """

        if self.n_values is None:
            return values

        if len(values) < self.n_values:
            full_vector = np.ones(self.n_values, dtype=values.dtype) * self.nan_value
            full_vector[: len(np.ravel(values))] = np.ravel(values)
            return full_vector

        if (
            len(values) > self.n_values
            and self.association is not DataAssociationEnum.OBJECT
        ):
            raise ValueError(
                f"Input 'values' of shape({self.n_values},) expected. "
                f"Array of shape{values.shape} provided.)"
            )

        return values

    def validate_values(self, values: np.ndarray | None) -> np.ndarray:
        """
        Check for possible mismatch between the length of values
        stored and the expected number of cells or vertices.

        :param values: Array of values to check

        :returns: values: An array of float values of length n_values or None
        """
        if values is None:
            return values

        if not isinstance(values, np.ndarray):
            raise TypeError("Input 'values' must be a numpy array.")

        if values.ndim > 1:
            values = np.ravel(values)
            warn("Input 'values' converted to a 1D array.")

        values = values.astype(float)

        # change nan values to nan_value
        values[np.isnan(values)] = self.nan_value

        # check the length of the values
        values = self.format_length(values)

        # check the value type
        values = self.format_type(values)

        return values
