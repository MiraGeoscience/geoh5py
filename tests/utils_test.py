# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2025 Mira Geoscience Ltd.                                     '
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

import numpy as np

from geoh5py.shared.utils import dip_azimuth_to_vector, find_unique_name


def test_find_unique_name():
    name = "test"
    names = ["test", "test(1)", "bidon"]

    assert find_unique_name(name, names) == "test(2)"


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
