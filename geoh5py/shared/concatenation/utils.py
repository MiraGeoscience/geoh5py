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

# pylint: disable=import-outside-toplevel, cyclic-import

from __future__ import annotations


def is_concatenator(object_) -> bool:
    """
    Check if the object is a Concatenator.

    :param object_: Object to check.
    :return: True if the object is a Concatenator.
    """
    from .concatenator import Concatenator

    return isinstance(object_, Concatenator)


def is_concatenated(object_) -> bool:
    """
    Check if the object is a Concatenated.

    :param object_: Object to check.
    :return: True if the object is a Concatenated.
    """
    from .concatenated import Concatenated

    return isinstance(object_, Concatenated)


def is_concatenated_object(object_) -> bool:
    """
    Check if the object is a ConcatenatedObject.

    :param object_: Object to check.
    :return: True if the object is a ConcatenatedObject.
    """
    from .object import ConcatenatedObject

    return isinstance(object_, ConcatenatedObject)


def is_concatenated_data(object_) -> bool:
    """
    Check if the object is a ConcatenatedData.

    :param object_: Object to check.
    :return: True if the object is a ConcatenatedData.
    """
    from .data import ConcatenatedData

    return isinstance(object_, ConcatenatedData)
