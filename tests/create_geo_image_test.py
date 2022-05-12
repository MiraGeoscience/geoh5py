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
import pytest

from geoh5py.objects import GeoImage
from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace


def test_create_copy_geoimage(tmp_path):

    workspace = Workspace(tmp_path / r"geo_image_test.geoh5")
    pixels = np.r_[
        np.c_[32, 0],
        np.c_[32, 64],
        np.c_[64, 64],
    ]
    points = np.r_[
        np.c_[5.0, 5.0, 0],
        np.c_[5.0, 10.0, 3],
        np.c_[10.0, 10.0, 3],
    ]
    geoimage = GeoImage.create(workspace, name="MyGeoImage")

    with pytest.raises(AttributeError) as excinfo:
        geoimage.georeference(pixels[0, :], points)

    assert "An 'image' must be set be" in str(excinfo)

    with pytest.raises(ValueError) as excinfo:
        geoimage.image = np.random.randn(12)

    assert (
        "Input 'value' for the 'image' property must be a 2D or 3D numpy.ndarray"
        in str(excinfo)
    )

    with pytest.raises(ValueError) as excinfo:
        geoimage.image = np.random.randn(12, 12, 4)

    assert (
        "Shape of the 'image' must be a 2D or a 3D array with shape(*,*, 3) "
        "representing 'RGB' values." in str(excinfo)
    )

    geoimage.image = np.random.randint(0, 255, (128, 128))

    with pytest.raises(ValueError) as excinfo:
        geoimage.georeference(pixels[0, :], points)

    assert (
        "Input reference points must be a 2D array of shape(*, 2) with at least 3 control points."
        in str(excinfo.value)
    )

    with pytest.raises(ValueError) as excinfo:
        geoimage.georeference(pixels, points[0, :])

    assert "Input 'locations' must be a 2D array of shape(*, 3)" in str(excinfo.value)

    geoimage.image = np.random.randint(0, 255, (128, 64, 3))
    geoimage.georeference(pixels, points)
    np.testing.assert_almost_equal(
        geoimage.vertices,
        np.asarray([[0, 15, 6], [10, 15, 6], [10, 5, 0], [0, 5, 0]]),
        err_msg="Issue geo-referencing the coordinates.",
    )

    geoimage_copy = GeoImage.create(workspace, name="MyGeoImageTwin")
    geoimage.image_data.copy(parent=geoimage_copy)

    np.testing.assert_almost_equal(geoimage_copy.vertices, geoimage.default_vertices)

    # Setting image from byte
    geoimage_copy = GeoImage.create(workspace, name="MyGeoImageTwin")
    geoimage_copy.image = geoimage.image_data.values
    assert geoimage_copy.image == geoimage.image, "Error setting image from bytes."

    # Re-load from file
    geoimage.image.save(tmp_path / r"test.tiff")
    geoimage_file = GeoImage.create(workspace, name="MyGeoImage")

    with pytest.raises(ValueError) as excinfo:
        geoimage_file.image = str(tmp_path / r"abc.tiff")

    assert "does not exist" in str(excinfo.value)

    geoimage_file.image = str(tmp_path / r"test.tiff")

    assert (
        geoimage_file.image == geoimage.image
    ), "Error writing and re-loading the image file."

    new_workspace = Workspace(tmp_path / r"geo_image_test2.geoh5")
    geoimage.copy(parent=new_workspace)

    new_workspace = Workspace(tmp_path / r"geo_image_test2.geoh5")
    rec_image = new_workspace.get_entity("MyGeoImage")[0]
    compare_entities(geoimage, rec_image, ignore=["_parent", "_image"])

    assert rec_image.image == geoimage.image, "Error copying the bytes image data."
