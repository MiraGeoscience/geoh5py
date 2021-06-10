#  Copyright (c) 2021 Mira Geoscience Ltd.
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

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from geoh5py.objects import Points
from geoh5py.workspace import Workspace

# from geoh5py.io.h5_writer import H5Writer


@patch("geoh5py.io.h5_writer.H5Writer.write_data_values")
@patch("geoh5py.io.h5_writer.H5Writer.write_coordinates")
@patch("geoh5py.io.h5_writer.H5Writer.write_attributes")
def test_save_modified_properties(
    write_attributes,
    write_coordinates,
    write_data_values,
):
    n_data = 12
    xyz = np.random.randn(n_data, 3)

    with tempfile.TemporaryDirectory() as tempdir:
        h5file_path = Path(tempdir) / r"testPoints.geoh5"

        # Create a workspace
        workspace = Workspace(h5file_path)
        points = Points.create(workspace)
        workspace.finalize()

        assert write_attributes.called, f"{write_attributes} was not called."
        assert (
            not write_coordinates.called
        ), f"{write_coordinates} should not have been called."
        assert (
            not write_data_values.called
        ), f"{write_data_values} should not have been called."

        points.vertices = xyz
        workspace.finalize()

        assert write_coordinates.called, f"{write_coordinates} should have been called."
        assert (
            not write_data_values.called
        ), f"{write_data_values} should not have been called."

        points.add_data({"rando": {"values": np.ones(n_data)}})

        assert write_data_values.called, f"{write_data_values} should have been called."
