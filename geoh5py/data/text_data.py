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

import json

import numpy as np
from numpy import ndarray

from ..shared.utils import as_str_if_uuid, dict_mapper
from .data import Data
from .primitive_type_enum import PrimitiveTypeEnum


def text_formating(values: None | np.ndarray | str) -> ndarray | None:
    """
    Format text values to utf-8.

    :param values: The values to format.

    :return: The formatted values.
    """
    # todo: values[0] seems dangerous here
    if values is None or isinstance(values[0], bytes):
        return values

    return np.char.encode(values, encoding="utf-8").astype("O")


class TextData(Data):
    @property
    def formatted_values(self):
        return text_formating(self.values)

    @property
    def nan_value(self):
        """
        Nan-Data-Value to be used in arrays
        """
        return ""

    @classmethod
    def primitive_type(cls) -> PrimitiveTypeEnum:
        return PrimitiveTypeEnum.TEXT

    def validate_values(
        self, values: np.ndarray | str | None
    ) -> np.ndarray | str | None:
        if isinstance(values, bytes):
            values = values.decode()

        if isinstance(values, np.ndarray) and values.dtype == object:
            values = np.array(
                [v.decode("utf-8") if isinstance(v, bytes) else v for v in values]
            )

        if (not isinstance(values, (str, type(None), np.ndarray))) or (
            isinstance(values, np.ndarray) and values.dtype.kind not in ["U", "S"]
        ):
            raise ValueError(
                f"Input 'values' for {self} must be of type {np.ndarray}  str or None."
            )

        return values


class CommentsData(Data):
    """
    Comments added to an Object or Group.
    Stored as a list of dictionaries with the following keys:

        .. code-block:: python

            comments = [
                {
                    "Author": "username",
                    "Date": "2020-05-21T10:12:15",
                    "Text": "A text comment."
                },
            ]
    """

    @property
    def formatted_values(self):
        return json.dumps(dict_mapper(self.values, [as_str_if_uuid]))

    @classmethod
    def primitive_type(cls) -> PrimitiveTypeEnum:
        return PrimitiveTypeEnum.TEXT

    def validate_values(self, values) -> dict | None:
        if isinstance(values, str):
            values = json.loads(values)

        if values is None:
            return None

        if not isinstance(values, dict):
            raise TypeError("Input 'values' for CommentsData must be a dict.")

        if "Comments" not in values:
            raise ValueError(
                "Input 'values' for CommentsData must contain key 'Comments'."
            )

        for value in values["Comments"]:
            if not isinstance(value, dict):
                raise TypeError(
                    "Error setting CommentsData with expected input of type list[dict].\n"
                    f"Input {type(values)} provided."
                )

            if not len(set(value.keys()).union({"Author", "Date", "Text"})) == 3:
                raise ValueError(
                    "Comment dictionaries must include keys 'Author', 'Date' and 'Text'.\n"
                    f"Keys {list(value.keys())} provided."
                )

        return values


class MultiTextData(Data):
    _values: np.ndarray | str | None

    @property
    def formatted_values(self):
        return text_formating(self.values)

    @classmethod
    def primitive_type(cls) -> PrimitiveTypeEnum:
        return PrimitiveTypeEnum.MULTI_TEXT

    def validate_values(
        self, values: np.ndarray | str | None
    ) -> np.ndarray | str | None:
        if not isinstance(values, np.ndarray | str | type(None)):
            raise ValueError(
                f"Input 'values' for {self} must be of type {np.ndarray}  str or None."
            )

        return values
