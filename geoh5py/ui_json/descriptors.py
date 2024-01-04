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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geoh5py.ui_json.forms import FormParameter
    from geoh5py.ui_json.parameters import Parameter


# pylint: disable=too-few-public-methods


class ValueAccess:
    """
    Descriptor to elevate underlying member values within 'FormParameter'.

    :param private: Name of private attribute.
    """

    def __init__(self, private: str):
        self.private: str = private

    def __get__(self, obj: Parameter, objtype=None):
        return getattr(obj, self.private).value

    def __set__(self, obj: Parameter, value):
        setattr(getattr(obj, self.private), "value", value)


class FormValueAccess:
    """
    Descriptor to elevate underlying member values within 'FormParameter'.

    :param private: Name of private attribute.
    """

    def __init__(self, private: str):
        self.private: str = private

    def __get__(self, obj: FormParameter, objtype=None):
        return getattr(obj, self.private).value

    def __set__(self, obj: FormParameter, value):
        setattr(getattr(obj, self.private), "value", value)
        obj._active_members.append(self.private[1:])
