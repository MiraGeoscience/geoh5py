#  Copyright (c) 2020 Mira Geoscience Ltd.
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

from typing import Optional

from numpy import array
from PIL.Image import Image  # pylint: disable=import-error

from .data import Data, PrimitiveTypeEnum


class FilenameData(Data):

    _file_object = None

    @classmethod
    def primitive_type(cls) -> PrimitiveTypeEnum:
        return PrimitiveTypeEnum.FILENAME

    @property
    def file_name(self) -> Optional[str]:
        """
        :obj:`str` Text value.
        """
        if (getattr(self, "_file_name", None) is None) and self.existing_h5_entity:
            self._file_name = self.workspace.fetch_values(self.uid)[0]

        return self._file_name

    @file_name.setter
    def file_name(self, value):
        self.modified_attributes = "values"
        self._file_name = value

    @property
    def file_object(self):
        if (
            self.file_name is not None
            and self.existing_h5_entity
            and getattr(self, "_file_object", None) is None
        ):
            self._file_object = self.workspace.fetch_file_object(
                self.uid, self.file_name
            )
        return self._file_object

    @property
    def values(self) -> Optional[str]:
        """
        :obj:`str` Text value.
        """
        if getattr(self, "_values", None) is None:
            if isinstance(self.file_object, Image):
                self._values = array(self._file_object)

        return self._values

    @values.setter
    def values(self, values):
        self.modified_attributes = "values"
        self._values = values

    def __call__(self):
        return self._file_name

    # TODO: implement specialization to access values.
    # Stored as a 1D array of 32-bit unsigned integer type (native).
    # Value map : 1D composite type array data set
    #   – Key (unsigned int)
    #   - Value (variable-length utf8 string)
    # must exist under type.
    # No data value : 0 (key is tied to value “Unknown”)
