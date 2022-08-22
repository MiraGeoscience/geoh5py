#  Copyright (c) 2022 Mira Geoscience Ltd.
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

from .data import Data, DataType, PrimitiveTypeEnum


class NumericData(Data, ABC):
    """
    Data container for floats values
    """

    def __init__(self, data_type: DataType, **kwargs):
        super().__init__(data_type, **kwargs)

    @classmethod
    def primitive_type(cls) -> PrimitiveTypeEnum:
        return PrimitiveTypeEnum.INVALID

    @property
    def ndv(self):
        """No-data-value"""
        ...

    @property
    def values(self) -> np.ndarray | None:
        """
        :return: values: An array of float values
        """
        if getattr(self, "_values", None) is None:
            values = self.workspace.fetch_values(self)

            if isinstance(values, (np.ndarray, type(None))):
                self._values = self.check_vector_length(values)

        return self._values

    @values.setter
    def values(self, values: np.ndarray | None):
        if isinstance(values, (np.ndarray, type(None))):
            values = self.check_vector_length(values)

        else:
            raise ValueError(
                f"Input 'values' for {self} must be of type {np.ndarray} or None."
            )

        self._values = values

        self.workspace.update_attribute(self, "values")

    def check_vector_length(self, values) -> np.ndarray:
        """
        Check for possible mismatch between the length of values
        stored and the expected number of cells or vertices.
        """
        if self.n_values is not None:
            if values is None or len(values) < self.n_values:
                full_vector = np.ones(self.n_values, dtype=type(self.ndv))
                if isinstance(self.ndv, float):
                    full_vector *= np.nan
                else:
                    full_vector *= self.ndv

                full_vector[: len(np.ravel(values))] = np.ravel(values)
                return full_vector

            if len(values) > self.n_values:
                raise ValueError(
                    f"Input 'values' of shape({self.n_values},) expected. "
                    f"Array of shape{values.shape} provided.)"
                )
        return values

    def __call__(self):
        return self.values
