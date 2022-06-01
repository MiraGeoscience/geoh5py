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


import numpy as np

from geoh5py.groups import DrillholeGroup
from geoh5py.objects import Drillhole
# from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace


# def test_create_drillhole_data():
#     h5file_path = r"C:\Users\dominiquef\Desktop\GA_demo_4.0.geoh5"
#
#     with Workspace(h5file_path) as w_s:
#         obj = w_s.groups[1].children[2]
#         data = obj.get_data("As")[0]
#         print(obj.get_data_list())
#         print(data.values)
#         print(obj.surveys)
def test_create_drillhole_data(tmp_path):
    h5file_path = tmp_path / r"testCurve.geoh5"
    well_name = "bullseye"
    n_data = 10
    collocation = 1e-5

    with Workspace(h5file_path) as workspace:
        # Create a workspace
        dh_group = DrillholeGroup.create(workspace)
        max_depth = 100
        Drillhole.create(
            workspace,
            collar=np.r_[0.0, 10.0, 10],
            surveys=np.c_[
                np.linspace(0, max_depth, n_data),
                np.linspace(-89, -75, n_data),
                np.ones(n_data) * 45.0,
            ],
            parent=dh_group,
            name=well_name,
            default_collocation_distance=collocation,
        )

        # value_map = {}
        # for ref in range(8):
        #     value_map[ref] = "".join(
        #         random.choice(string.ascii_lowercase) for i in range(8)
        #     )
        #
        # # Create random from-to
        # from_to_a = np.sort(
        #     np.random.uniform(low=0.05, high=max_depth, size=(50,))
        # ).reshape((-1, 2))
        # from_to_b = np.vstack([from_to_a[0, :], [30.1, 55.5], [56.5, 80.2]])
        #
        # # Add from-to data
        # data_objects = well.add_data(
        #     {
        #         "interval_values": {
        #             "values": np.random.randn(from_to_a.shape[0]),
        #             "from-to": from_to_a,
        #         },
        #         "int_interval_list": {
        #             "values": [1, 2, 3],
        #             "from-to": from_to_b,
        #             "value_map": {1: "Unit_A", 2: "Unit_B", 3: "Unit_C"},
        #             "type": "referenced",
        #         },
        #     }
        # )
