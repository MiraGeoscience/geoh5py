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

import numpy as np
from h5py import File

from geoh5py.objects import Points
from geoh5py.shared import fetch_h5_handle
from geoh5py.workspace import Workspace


def test_scope_write(tmp_path):
    with fetch_h5_handle(os.path.join(tmp_path, "test2.geoh5"), mode="a") as h5file:
        w_s = Workspace(h5file)
        xyz = np.array([[1, 2, 3], [4, 5, 6]])
        for _ in range(10):
            Points.create(w_s, vertices=xyz)

    Points.create(w_s, vertices=xyz)


def test_fetch_handle(tmp_path):
    h5file_path = os.path.join(tmp_path, "test2.geoh5")
    w_s = Workspace(h5file_path)
    with File(h5file_path, "r+") as project:
        base = list(project.keys())[0]
        del project[base]["Objects"]
        del project[base]["Types"]

    Points.create(w_s)
