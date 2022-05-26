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


# import numpy as np
#
# from geoh5py.objects import Drillhole
# from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace


def test_create_drillhole_data(tmp_path):
    h5file_path = r"C:\Users\dominiquef\Desktop\GA_demo_4.0.geoh5"

    with Workspace(h5file_path) as w_s:
        obj = w_s.objects[0]
        obj.get_data_values("As")
