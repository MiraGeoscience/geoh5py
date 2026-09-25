# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2020-2026 Mira Geoscience Ltd.                                '
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

from pathlib import Path
from typing import Any, cast
from warnings import warn

import numpy as np

from .data import Data
from .data_association_enum import DataAssociationEnum


class FilenameData(Data):
    """
    Class for storing files as data blob.

    :param values: Name of the file.
    :param file_bytes: Binary representation of the file.
    """

    def __init__(
        self,
        values: str | None = None,
        file_bytes: dict[str, bytes] | None = None,
        name="GeoImageMesh_Image",
        public: bool = False,
        **kwargs,
    ):
        self._file_bytes = None

        super().__init__(values=values, name=name, public=public, **kwargs)

        self.file_bytes = file_bytes

    @property
    def file_bytes(self):
        """
        Binary blob value representation of a file.
        """
        if self.values is not None and self.on_file and self._file_bytes is None:
            file_bytes = {}
            for value in self.values:
                byte_data = self.workspace.fetch_file_object(self, value)

                if byte_data is not None:
                    file_bytes[value] = byte_data

            self._file_bytes = cast(dict[str, bytes], file_bytes)

        return self._file_bytes

    @file_bytes.setter
    def file_bytes(self, value: bytes | dict[str, bytes] | None):
        if value is not None and self.values is None:
            raise AttributeError("FilenameData requires the 'values' to be set.")

        if isinstance(value, dict):
            if not all(
                isinstance(k, str) and isinstance(v, bytes) for k, v in value.items()
            ):
                raise TypeError(
                    "Input 'file_bytes' for FilenameData must be a dict of "
                    "string keys and bytes values."
                )

        elif value is not None:
            if not isinstance(value, bytes):
                raise TypeError(
                    "Input 'file_bytes' for FilenameData must be of type 'bytes'."
                )
            value = {self.values[0]: value}

        self._file_bytes = value

        if self._file_bytes is not None and self.on_file:
            self.workspace.update_attribute(self, "values")

    @property
    def file_name(self):
        """
        Binary blob value representation of a file.
        """
        warn("This method is deprecated. Use 'values' instead.", DeprecationWarning)

        return self.values

    def save_file(self, path: str | Path = Path()) -> Path:
        """
        Save the file to disk.

        :param path: Directory to save the file to.

        :return: Path to the saved file.
        """
        Path(path).mkdir(exist_ok=True)

        if self.file_bytes is not None:
            for name, file_bytes in self.file_bytes.items():
                with open(Path(path) / name, "wb") as raw_binary:
                    raw_binary.write(file_bytes)

        return Path(path)

    def validate_values(self, values: Any | None) -> np.ndarray[str]:

        if values is None:
            return values

        if isinstance(values, str):
            values = np.asarray(values, dtype=str).reshape((1,))

        if not (
            isinstance(values, np.ndarray) and np.issubdtype(values.dtype, np.flexible)
        ):
            raise TypeError(
                "Input 'values' for FilenameData must be of type 'np.ndarray' with string dtype."
            )

        if self.association is DataAssociationEnum.OBJECT and len(values) != 1:
            raise ValueError(
                "Input 'values' for FilenameData with OBJECT association must be a single string."
            )

        if (
            self.association is DataAssociationEnum.VERTEX
            and len(values) != self.parent.n_vertices
        ):
            raise ValueError(
                "Input 'values' for FilenameData with VERTEX association must have the "
                "same length as the number of vertices in the parent object."
            )

        return values
