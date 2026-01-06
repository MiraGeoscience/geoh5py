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

from uuid import uuid4

import numpy as np
import pytest

from geoh5py import Workspace
from geoh5py.objects import Points
from geoh5py.shared.utils import (
    copy_dict_relatives,
    dip_azimuth_to_vector,
    extract_uids,
    find_unique_name,
    format_numeric_values,
    split_name_suffixes,
)


def test_find_unique_name():
    name = "test"
    names = ["test", "test(1)", "bidon"]

    assert find_unique_name(name, names) == "test(2)"

    name = "test(1)"

    assert find_unique_name(name, names) == "test(2)"


def test_split_name_suffix():
    name = "test.ui.json"
    assert split_name_suffixes(name) == ("test", ".ui.json")

    name = "test"

    assert split_name_suffixes(name) == ("test", "")


def test_find_unique_name_files():
    name = "test.ui.json"
    names = ["test.ui.json", "test(1).ui.json", "bidon"]

    assert find_unique_name(name, names) == "test(2).ui.json"

    name = "test(1).ui.json"

    assert find_unique_name(name, names) == "test(2).ui.json"

    name = "Test(1).ui.json"

    assert find_unique_name(name, names, case_sensitive=False) == "Test(2).ui.json"


def test_dip_azimuth_to_vector():
    dip = -45
    azimuth = 90

    np.testing.assert_almost_equal(
        dip_azimuth_to_vector(dip, azimuth), np.c_[0.7071, 0.0, -0.7071], decimal=3
    )

    dip = 45
    azimuth = 245

    np.testing.assert_almost_equal(
        dip_azimuth_to_vector(dip, azimuth), np.c_[-0.641, -0.299, 0.707], decimal=3
    )


def test_format_values_for_display():
    test = np.array(
        [1145.199999, 0.01452145, 0.001452145, 741254125.74125, 0.0000145, np.nan]
    )

    expected = ["1145.2", "0.014521", "1.45214e-03", "7.41254e+08", "1.45e-05", ""]
    result = format_numeric_values(test, 5, 8).tolist()

    assert result == expected


def test_extract_uids_errors():
    assert extract_uids(None) is None

    with pytest.raises(TypeError, match="'bidon' must be of type"):
        extract_uids("bidon")

    class Bidon:
        def __init__(self, uid):
            self.uid = uid

    uid = uuid4()

    assert extract_uids(Bidon(uid)) == [uid]  # type: ignore


def test_copy_relatives_errors():
    workspace = Workspace()
    points = Points.create(workspace, name="points", vertices=np.random.rand(10, 3))

    with pytest.raises(ValueError, match="Cannot copy "):
        copy_dict_relatives({"bidon": points}, workspace)
