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

from .data import Data, PrimitiveTypeEnum
from .data_type import DataType


class FilenameData(Data):
    _file_name: str | None = None
    _name = "GeoImageMesh_Image"
    _values: bytes | None

    def __init__(self, data_type: DataType, file_name=None, **kwargs):
        super().__init__(data_type, file_name=file_name, **kwargs)
        self._public = False

    @classmethod
    def primitive_type(cls) -> PrimitiveTypeEnum:
        return PrimitiveTypeEnum.FILENAME

    @property
    def file_name(self) -> str | None:
        """
        :obj:`str` Text value.
        """
        if getattr(self, "_file_name", None) is None:
            file_name = self.workspace.fetch_values(self)

            if isinstance(file_name, (str, type(None))):
                self._file_name = file_name

        return self._file_name

    @file_name.setter
    def file_name(self, value: str | None):
        self._file_name = value

        if not isinstance(value, (str, type(None))):
            raise ValueError(
                f"Input 'file_name' for {self} must be of type str or None."
            )

        self.workspace.update_attribute(self, "values")

    def save_file(self, path: str | Path = Path(), name=None):
        """
        Save the file to disk.

        :param path: Directory to save the file to.
        :param name: Name given to the file.
        """
        Path(path).mkdir(exist_ok=True)
        if name is None:
            name = getattr(self, "file_name", "image.tiff")

        if self.values is not None:
            with open(Path(path) / name, "wb") as raw_binary:
                raw_binary.write(self.values)

    @property
    def values(self) -> bytes | None:
        """
        Binary :obj:`str` value representation of a file.
        """
        if (
            self.file_name is not None
            and self.on_file
            and getattr(self, "_values", None) is None
        ):
            self._values = self.workspace.fetch_file_object(self.uid, self.file_name)

        return self._values

    @values.setter
    def values(self, values):
        if not isinstance(values, bytes):
            raise ValueError("Input 'values' for FilenameData must be of type 'bytes'.")

        if self.file_name is None:
            raise AttributeError("FilenameData requires the 'file_name' to be set.")

        self._values = values
        self.workspace.update_attribute(self, "values")

    # TODO: implement specialization to access values.
    # Stored as a 1D array of 32-bit unsigned integer type (native).
    # Value map : 1D composite type array data set
    #   – Key (unsigned int)
    #   - Value (variable-length utf8 string)
    # must exist under type.
    # No data value : 0 (key is tied to value “Unknown”)
