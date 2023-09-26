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

from typing import Any

from .input_file import InputFile
from .utils import monitored_directory_copy
from .validation import InputValidation


class SetDict(dict):
    """
    A dictionary that stores a collection of unique values.

    :Note: Setting replaces, but an update results in the union of
         any existing and update sets.
    """

    def __init__(self, **kwargs):
        self.update(kwargs)

    def update(self, data: dict[str, Any]):  # type: ignore
        """Update keys and associated sets."""

        for key, value in data.items():
            if key in self.__dict__:
                self.__dict__[key].update(self._make_set(value))
            else:
                self.__setitem__(key, value)

    def _make_set(self, value) -> set:
        """Converts value to a set."""
        if not hasattr(value, "__iter__") or isinstance(value, (str, type)):
            value = [value]
        return set(value)

    def __setitem__(self, key: str, item):
        self.__dict__[key] = self._make_set(item)

    def __getitem__(self, key: str) -> set:
        return self.__dict__[key]

    def __repr__(self):
        return repr(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __contains__(self, item):
        return item in self.__dict__

    def __iter__(self):
        return iter(self.__dict__)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def pop(self, *args):
        return self.__dict__.pop(*args)
