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

import numpy as np

from geoh5py.objects import Drillhole
from geoh5py.workspace import Workspace


def test_insert_drillhole_data(tmp_path):
    well_name = "bullseye"
    n_data = 10
    collocation = 1e-5
    h5file_path = tmp_path / r"testCurve.geoh5"

    with Workspace(version=1.0).save_as(h5file_path) as workspace:
        max_depth = 100
        well = Drillhole.create(
            workspace,
            collar=np.r_[0.0, 10.0, 10],
            surveys=np.c_[
                np.linspace(0, max_depth, n_data),
                np.linspace(-89, -75, n_data),
                np.ones(n_data) * 45.0,
            ],
            name=well_name,
            default_collocation_distance=collocation,
        )
        # Add log-data
        data_object = well.add_data(
            {
                "log_values": {
                    "depth": np.sort(np.random.rand(n_data) * max_depth),
                    "values": np.random.randint(1, high=8, size=n_data),
                }
            }
        )

        # Add more data with single match
        old_depths = well.get_data("DEPTH")[0].values
        indices = np.where(~np.isnan(old_depths))[0]
        insert = np.random.randint(0, high=len(indices) - 1, size=2)
        new_depths = old_depths[indices[insert]]
        new_depths[0] -= 2e-6  # Out of tolerance
        new_depths[1] -= 5e-7  # Within tolerance

        match_test = well.add_data(
            {
                "match_depth": {
                    "depth": new_depths,
                    "values": np.random.randint(1, high=8, size=2),
                    "collocation_distance": 1e-6,
                }
            }
        )

        assert well.n_vertices == n_data + 1, (
            "Error adding values with collocated tolerance"
        )
        assert np.isnan(data_object.values[indices[insert][0]]), (
            "Old values not re-sorted properly after insertion"
        )

        insert_ind = np.where(~np.isnan(match_test.values))[0]
        if insert[0] <= insert[1]:
            assert all(
                ind in [indices[insert][0], indices[insert][1] + 1]
                for ind in insert_ind
            ), "Depth insertion error"
        else:
            assert all(
                ind in [indices[insert][0], indices[insert][1]] for ind in insert_ind
            ), "Depth insertion error"
