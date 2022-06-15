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

import random
import string
import numpy as np

from geoh5py.groups import DrillholeGroup
from geoh5py.objects import Drillhole
from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace


def test_load_drillhole_data():
    h5file_path = r"C:\Users\dominiquef\Desktop\testCurve - Copy.geoh5"

    ws_1 = Workspace(h5file_path)
    # ws_2 = Workspace(
    #     r"C:\Users\dominiquef\AppData\Local\Temp\pytest-of-dominiquef\pytest-123\test_create_drillhole_data0\testCurve.geoh5"
    # )

    good = ws_1.get_entity("Entity")[0]
    good.children[0].surveys
    # bad = ws_2.get_entity("Drillholes Group")[0]
    # compare_entities(good, bad, ignore=["_uid", "_parent", "_description", "_name", "_on_file"])
    # with Workspace(h5file_path) as w_s:
    #     group = w_s.groups[1]
    #     obj = group.children[0]
    #     print(obj.property_groups)
    #     obj.collar = [314000.0, 6075000.0, 200.0]
    #     # data = obj.get_data("As")[0]
    #     data.values = data.values**0.0
    #
    #     # print(obj.get_data_list())
    #     # print(data.values)
    #     assert obj.surveys is not None


def test_create_drillhole_data(tmp_path):
    h5file_path = tmp_path / r"testCurve.geoh5"
    well_name = "bullseye"
    n_data = 10
    collocation = 1e-5

    with Workspace(h5file_path, version=2.0) as workspace:
        # Create a workspace
        dh_group = DrillholeGroup.create(workspace)
        max_depth = 100
        well = Drillhole.create(
            workspace,
            collar=np.r_[0.0, 10.0, 10],
            surveys=np.c_[
                np.linspace(0, max_depth, n_data),
                np.ones(n_data) * 45.0,
                np.linspace(-89, -75, n_data)
            ],
            parent=dh_group,
            name=well_name,
            default_collocation_distance=collocation,
        )

        value_map = {}
        for ref in range(8):
            value_map[ref] = "".join(
                random.choice(string.ascii_lowercase) for i in range(8)
            )

        # Create random from-to
        from_to_a = np.sort(
            np.random.uniform(low=0.05, high=max_depth, size=(50,))
        ).reshape((-1, 2))
        from_to_b = np.vstack([from_to_a[0, :], [30.1, 55.5], [56.5, 80.2]])

        # Add from-to data
        data_objects = well.add_data(
            {
                "interval_values": {
                    "values": np.random.randn(from_to_a.shape[0]),
                    "from-to": from_to_a,
                },
                "int_interval_list": {
                    "values": [1, 2, 3],
                    "from-to": from_to_b,
                    "value_map": {1: "Unit_A", 2: "Unit_B", 3: "Unit_C"},
                    "type": "referenced",
                },
            }
        )
