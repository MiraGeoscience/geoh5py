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

from os import path

import numpy as np

from geoh5py.objects import GeoImage
from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace


def test_create_copy_geoimage(tmp_path):

    workspace = Workspace(path.join(tmp_path, "geo_image_test2.geoh5"))

    values = np.random.randint(0, 255, (128, 64, 3))

    points = np.r_[
        np.c_[5.0, 5.0, 0],
        np.c_[5.0, 10.0, 3],
        np.c_[10.0, 10.0, 3],
    ]
    pixels = np.r_[
        np.c_[32, 0],
        np.c_[32, 64],
        np.c_[64, 64],
    ]

    geoimage = GeoImage.create(workspace, name="MyGeoImage")
    geoimage.image = values
    geoimage.georeference(pixels, points)
    workspace.finalize()

    np.testing.assert_almost_equal(
        geoimage.vertices[:, 0].max(),
        20,
        err_msg="Issue geo-referencing the max x-coordinates.",
    )
    np.testing.assert_almost_equal(
        geoimage.vertices[:, 0].min(),
        0,
        err_msg="Issue geo-referencing the min x-coordinates.",
    )

    np.testing.assert_almost_equal(
        geoimage.vertices[:, 1].max(),
        10,
        err_msg="Issue geo-referencing the max y-coordinates.",
    )
    np.testing.assert_almost_equal(
        geoimage.vertices[:, 1].min(),
        5,
        err_msg="Issue geo-referencing the min y-coordinates.",
    )

    np.testing.assert_almost_equal(
        geoimage.vertices[:, 2].max(),
        3,
        err_msg="Issue geo-referencing the max z-coordinates.",
    )
    np.testing.assert_almost_equal(
        geoimage.vertices[:, 2].min(),
        0,
        err_msg="Issue geo-referencing the min z-coordinates.",
    )

    new_workspace = Workspace(path.join(tmp_path, "geo_image_test2.geoh5"))
    geoimage.copy(parent=new_workspace)
    rec_image = new_workspace.get_entity("MyGeoImage")[0]

    compare_entities(geoimage, rec_image, ignore=["_parent", "_image"])

    assert rec_image.image == geoimage.image, "Error copying the bytes image data."
