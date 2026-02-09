# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2020-2026 Mira Geoscience Ltd.                                     '
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

import weakref

import pytest

from geoh5py.shared import weakref_utils


class AnyObject:
    # pylint: disable=too-few-public-methods
    pass


def test_remove_none_referents():
    some_dict = {"gone": weakref.ref(AnyObject())}
    bound_object = AnyObject()
    some_dict["there"] = weakref.ref(bound_object)

    assert "gone" in some_dict
    assert some_dict["gone"]() is None
    assert some_dict["there"]() is bound_object
    weakref_utils.remove_none_referents(some_dict)
    assert "gone" not in some_dict


def test_get_clean_ref():
    some_dict = {"gone": weakref.ref(AnyObject())}
    bound_object = AnyObject()
    some_dict["there"] = weakref.ref(bound_object)

    assert "gone" in some_dict
    assert some_dict["gone"]() is None
    assert some_dict["there"]() is bound_object
    assert weakref_utils.get_clean_ref(some_dict, "there") is bound_object
    assert weakref_utils.get_clean_ref(some_dict, "gone") is None
    assert "gone" not in some_dict
    assert "there" in some_dict
    assert some_dict["there"]() is bound_object


def test_insert_once():
    some_dict = {"gone": weakref.ref(AnyObject())}
    bound_object = AnyObject()
    some_dict["there"] = weakref.ref(bound_object)

    assert "gone" in some_dict
    assert some_dict["gone"]() is None
    assert some_dict["there"]() is bound_object

    other = AnyObject()
    with pytest.raises(RuntimeError) as error:
        weakref_utils.insert_once(some_dict, "there", other)
    assert "Key 'there' already used" in str(error.value)

    weakref_utils.insert_once(some_dict, "gone", other)
    assert some_dict["gone"]() is other
