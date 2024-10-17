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

from pathlib import Path
from typing import Any
from warnings import warn

from .data import Data, PrimitiveTypeEnum


class FilenameData(Data):
    """
    Class for storing files as data blob.

    :param values: Name of the file.
    :param file_bytes: Binary representation of the file.
    """

    def __init__(
        self,
        values: str | None = None,
        file_bytes: bytes | None = None,
        name="GeoImageMesh_Image",
        public: bool = False,
        **kwargs,
    ):
        super().__init__(values=values, name=name, public=public, **kwargs)

        self.file_bytes = file_bytes

    @classmethod
    def primitive_type(cls) -> PrimitiveTypeEnum:
        return PrimitiveTypeEnum.FILENAME

    @property
    def file_bytes(self):
        """
        Binary blob value representation of a file.
        """
        if (
            self.values is not None
            and self.on_file
            and getattr(self, "_file_bytes", None) is None
        ):
            self._file_bytes = self.workspace.fetch_file_object(self.uid, self.values)

        return self._file_bytes

    @file_bytes.setter
    def file_bytes(self, value: bytes | None):
        if value is not None and self.values is None:
            raise AttributeError("FilenameData requires the 'values' to be set.")

        if not isinstance(value, bytes | None):
            raise TypeError(
                "Input 'file_bytes' for FilenameData must be of type 'bytes'."
            )

        self._file_bytes = value

        if self.on_file:
            self.workspace.update_attribute(self, "values")

    @property
    def file_name(self):
        """
        Binary blob value representation of a file.
        """
        warn("This method is deprecated. Use 'values' instead.", DeprecationWarning)

        return self.values

    def save_file(self, path: str | Path = Path(), name=None):
        """
        Save the file to disk.

        :param path: Directory to save the file to.
        :param name: Name given to the file.
        """
        Path(path).mkdir(exist_ok=True)
        if name is None:
            name = getattr(self, "values", "image.tiff")

        if self.file_bytes is not None:
            with open(Path(path) / name, "wb") as raw_binary:
                raw_binary.write(self.file_bytes)

    def validate_values(self, values: Any | None) -> Any:
        if not isinstance(values, str | None):
            raise ValueError("Input 'values' for FilenameData must be of type 'str'.")

        return values

    # TODO: implement specialization to access values.
    # Stored as a 1D array of 32-bit unsigned integer type (native).
    # Value map : 1D composite type array data set
    #   – Key (unsigned int)
    #   - Value (variable-length utf8 string)
    # must exist under type.
    # No data value : 0 (key is tied to value "Unknown")
