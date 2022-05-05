#  Copyright (c) 2022 Mira Geoscience Ltd.
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

import os
from time import time

import numpy as np

from geoh5py.objects import Points
from geoh5py.shared import fetch_h5_handle
from geoh5py.workspace import Workspace


def test_scope_write(tmp_path):
    ctime = time()
    w_s = Workspace(os.path.join(tmp_path, "test.geoh5"))
    xyz = np.array([[1, 2, 3], [4, 5, 6]])
    for _ in range(10):
        Points.create(w_s, vertices=xyz)

    runtime_1 = time() - ctime

    ctime = time()
    with fetch_h5_handle(os.path.join(tmp_path, "test2.geoh5"), mode="a") as h5file:
        w_s = Workspace(h5file)
        xyz = np.array([[1, 2, 3], [4, 5, 6]])
        for _ in range(10):
            Points.create(w_s, vertices=xyz)

    runtime_2 = time() - ctime

    assert runtime_2 < runtime_1, "Scope method should be faster."

    Points.create(w_s, vertices=xyz)
