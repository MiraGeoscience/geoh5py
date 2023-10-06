#  Copyright (c) 2023 Mira Geoscience Ltd.
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

# pylint: disable=unused-import
# flake8: noqa

from __future__ import annotations

from .input_file import InputFile
from .utils import monitored_directory_copy
from .validation import InputValidation


class SetDict(dict):
    def __init__(self, **kwargs):
        kwargs = {k: self.make_set(v) for k, v in kwargs.items()}
        super().__init__(kwargs)

    def make_set(self, value):
        if isinstance(value, (set, tuple, list)):
            value = set(value)
        else:
            value = {value}
        return value

    def __setitem__(self, key, value):
        value = self.make_set(value)
        super().__setitem__(key, value)

    def update(self, value: dict, **kwargs) -> None:  # type: ignore
        for key, val in value.items():
            val = self.make_set(val)
            if key in self:
                val = self[key].union(val)
            value[key] = val
        super().update(value, **kwargs)
