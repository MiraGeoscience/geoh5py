#  Copyright (c) 2020 Mira Geoscience Ltd.
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

import numpy as np
from scipy import spatial

from geoh5py.objects import Surface
from geoh5py.workspace import Workspace


def test_create_surface_data():

    with tempfile.TemporaryDirectory() as tempdir:
        h5file_path = Path(tempdir) / r"testSurface.geoh5"

        workspace = Workspace(h5file_path)

        # Create a grid of points and triangulate
        x, y = np.meshgrid(np.arange(10), np.arange(10))
        x, y = x.ravel(), y.ravel()
        z = np.random.randn(x.shape[0])

        del_surf = spatial.Delaunay(np.c_[x, y])

        simplices = getattr(del_surf, "simplices")

        # Create random data
        values = np.mean(
            np.c_[x[simplices[:, 0]], x[simplices[:, 1]], x[simplices[:, 2]]], axis=1
        )

        # Create a geoh5 surface
        surface = Surface.create(
            workspace, name="mySurf", vertices=np.c_[x, y, z], cells=simplices
        )

        data = surface.add_data({"TMI": {"values": values}})

        # Read the object from a different workspace object on the same file
        new_workspace = Workspace(h5file_path)

        obj_copy = new_workspace.get_entity("mySurf")[0]
        data_copy = obj_copy.get_data("TMI")[0]

        assert [
            prop in obj_copy.get_data_list() for prop in surface.get_data_list()
        ], "The surface object did not copy"
        assert np.all(data_copy.values == data.values), "Data values were not copied"
