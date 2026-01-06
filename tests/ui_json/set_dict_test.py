# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2025-2026 Mira Geoscience Ltd.                                '
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

from geoh5py.shared.utils import SetDict


def test_dict_set_class():
    test = SetDict(a=1, b=[2, 3])
    assert repr(test) == "{'a': {1}, 'b': {2, 3}}"
    assert test["a"] == {1}
    test["a"] = [1, 2]
    assert test["a"] == {1, 2}
    test["a"] = {1, 2}
    assert test["a"] == {1, 2}
    test.update({"b": 4})
    assert test["b"] == {2, 3, 4}
    test.update({"c": "hello"})
    assert test["c"] == {"hello"}
    for v in test.values():  # pylint: disable=invalid-name
        assert isinstance(v, set)
    assert len(test) == 3
    assert list(test) == ["a", "b", "c"]
    assert repr(test) == "{'a': {1, 2}, 'b': {2, 3, 4}, 'c': {'hello'}}"
    assert test
    test = SetDict()
    assert not test
