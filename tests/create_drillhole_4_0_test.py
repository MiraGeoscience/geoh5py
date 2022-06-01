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


def test_create_drillhole_data():
    h5file_path = r"C:\Users\dominiquef\Desktop\GA_demo_4.0.geoh5"

    with Workspace(h5file_path) as w_s:
        obj = w_s.groups[1].children[2]
        data = obj.get_data("As")[0]
        print(obj.get_data_list())
        print(data.values)
        print(obj.surveys)
