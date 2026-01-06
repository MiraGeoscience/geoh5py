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

from unittest.mock import patch

import numpy as np

from geoh5py.objects import Points
from geoh5py.workspace import Workspace


@patch("geoh5py.io.h5_writer.H5Writer.write_data_values")
@patch("geoh5py.io.h5_writer.H5Writer.write_array_attribute")
@patch("geoh5py.io.h5_writer.H5Writer.write_attributes")
def test_save_modified_properties(
    write_attributes, write_array_attribute, write_data_values, tmp_path
):
    n_data = 12
    xyz = np.random.randn(n_data, 3)

    h5file_path = tmp_path / r"testPoints.geoh5"

    # Create a workspace
    with Workspace.create(h5file_path) as workspace:
        points = Points.create(workspace, vertices=xyz)

        assert write_attributes.called, f"{write_attributes} was not called."
        assert write_array_attribute.called, f"{write_array_attribute} was not called."
        assert not write_data_values.called, (
            f"{write_data_values} should not have been called."
        )

        assert not write_data_values.called, (
            f"{write_data_values} should not have been called."
        )

        points.add_data({"rando": {"values": np.ones(n_data)}})

        assert write_data_values.called, f"{write_data_values} should have been called."
