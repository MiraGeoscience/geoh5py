#  Copyright (c) 2023 Mira Geoscience Ltd.
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


from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from PIL.TiffImagePlugin import TiffImageFile

from geoh5py.objects import GeoImage, Grid2D
from geoh5py.shared.conversion import GeoImageConversion
from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace

# test tag
tag = {
    256: (128,),
    257: (128,),
    258: (8, 8, 8),
    259: (1,),
    33922: (0.0, 0.0, 0.0, 522796.33210329525, 7244067.563364625, 0.0),
    42113: ("255",),
    262: (2,),
    33550: (0.9990415797117552, 0.999041579711816, 0.0),
    339: (1, 1, 1),
    277: (3,),
    278: (5,),
    284: (1,),
    34737: ("WGS 84 / UTM zone 34N|WGS 84|",),
}


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

    assert geoimage is not None

    assert geoimage.default_vertices is None

    assert geoimage.image_georeferenced is None

    with pytest.raises(AttributeError, match="The object contains no image data"):
        geoimage.save_as("test")

    with pytest.raises(AttributeError, match="An 'image' must be set be"):
        geoimage.georeference(pixels[0, :], points)

    with pytest.raises(ValueError, match="Input 'vertices' must be"):
        geoimage.vertices = [1, 2, 3]

    with pytest.raises(
        ValueError,
        match="Input 'value' for the 'image' property must be a 2D or 3D numpy.ndarray",
    ):
        geoimage.image = np.random.randn(12)

    with pytest.raises(ValueError, match="Shape of the 'image' must be a 2D or "):
        geoimage.image = np.random.randn(12, 12, 4)

    with pytest.raises(AttributeError, match="GeoImage has no vertices"):
        geoimage.to_grid2d()

    assert geoimage.image is None

    with pytest.raises(AttributeError, match="There is no image to"):
        geoimage.set_tag_from_vertices()

    with pytest.raises(AttributeError, match="The image is not georeferenced"):
        geoimage.georeferencing_from_tiff()

    geoimage.image = np.random.randint(0, 255, (128, 128))

    # with pytest.raises(AttributeError, match="Vertices must be set for referencing"):
    #     geoimage.set_tag_from_vertices()

    with pytest.raises(ValueError, match="Input reference points must be a 2D array"):
        geoimage.georeference(pixels[0, :], points)

    with pytest.raises(
        ValueError, match="Input 'locations' must be a 2D array of shape"
    ):
        geoimage.georeference(pixels, points[0, :])

    geoimage.image = np.random.randint(0, 255, (128, 64, 3))
    geoimage.georeference(pixels, points)
    np.testing.assert_almost_equal(
        geoimage.vertices,
        np.asarray([[0, 15, 6], [10, 15, 6], [10, 5, 0], [0, 5, 0]]),
        err_msg="Issue geo-referencing the coordinates.",
    )

    geoimage.to_grid2d()
    geoimage.save_as("testtif.tif", str(tmp_path))

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

    with pytest.raises(ValueError, match="does not exist"):
        geoimage_file.image = str(tmp_path / r"abc.tiff")

    geoimage_file.image = str(tmp_path / r"test.tiff")

    assert (
        geoimage_file.image == geoimage.image
    ), "Error writing and re-loading the image file."

    new_workspace = Workspace(tmp_path / r"geo_image_test2.geoh5")
    geoimage.copy(parent=new_workspace)

    new_workspace = Workspace(tmp_path / r"geo_image_test2.geoh5")
    rec_image = new_workspace.get_entity("MyGeoImage")[0]

    compare_entities(geoimage, rec_image, ignore=["_parent", "_image", "_tag"])

    assert rec_image.image == geoimage.image, "Error copying the bytes image data."

    geoimage.vertices = geoimage.vertices

    # Test copy from extent that clips one corner
    new_image = geoimage.copy_from_extent(np.vstack([[9, 9], [10, 10]]))
    assert new_image is not None, "Error copying from extent."

    new_image = geoimage.copy_from_extent(np.vstack([[100, 100], [200, 200]]))
    assert new_image is None, "Error copying from extent that is out of bounds."


def test_georeference_image(tmp_path):
    workspace = Workspace(tmp_path / r"geo_image_test.geoh5")

    # create and save a tiff
    image = Image.fromarray(
        np.random.randint(0, 255, (128, 128, 3)).astype("uint8"), "RGB"
    )
    for id_ in tag.items():
        image.getexif()[id_[0]] = id_[1]
    image.save(tmp_path / r"testtif.tif", exif=image.getexif())

    # load image
    geoimage = GeoImage.create(
        workspace, name="test_area", image=f"{str(tmp_path)}/testtif.tif"
    )

    geoimage.tag = None

    # test grid2d errors
    with pytest.raises(ValueError, match="Input 'tag' must"):
        geoimage.tag = 42

    # image = Image.open(tmp_path / r"testtif.tif")
    geoimage.tag = {"test": 3}
    geoimage.georeferencing_from_tiff()

    image = Image.open(f"{str(tmp_path)}/testtif.tif")

    geoimage = GeoImage.create(workspace, name="test_area", image=image)

    # create Gray grid2d
    grid2d_gray = geoimage.to_grid2d()

    # create RGB grid2d
    grid2d_rgb = geoimage.to_grid2d(new_name="RGB", transform="RGB")

    assert isinstance(grid2d_gray, Grid2D)
    assert isinstance(grid2d_rgb, Grid2D)
    assert isinstance(geoimage.image_georeferenced, Image.Image)

    # test grid2d errors
    with pytest.raises(KeyError, match="has to be 'GRAY"):
        geoimage.to_grid2d(new_name="RGB", transform="bidon")

    # test save_as
    with pytest.raises(TypeError, match="has to be a string"):
        geoimage.save_as(0)

    with pytest.raises(TypeError, match="has to be a string"):
        geoimage.save_as("test", 0)

    with pytest.raises(FileNotFoundError, match="No such file or directory"):
        geoimage.save_as("test", "path/bidon")

    geoimage.save_as("saved_tif.tif", str(tmp_path))
    image = Image.open(tmp_path / r"saved_tif.tif")

    assert isinstance(image, TiffImageFile)

    geoimage.save_as("saved_tif.png", str(tmp_path))

    image = Image.open(f"{str(tmp_path)}/testtif.tif").convert("L")
    geoimage = GeoImage.create(workspace, name="test_area", image=image)

    # test grid2d errors
    with pytest.raises(IndexError, match="have 3 bands"):
        geoimage.to_grid2d(new_name="RGB", transform="RGB")

    # extensive test conversion
    # with pytest.raises(TypeError, match="Entity must be a 'GeoImage'"):
    #     _ = GeoImageConversion(["bidon"])

    converter = GeoImageConversion
    # geoimage.conversion_type.to_grid2d(geomiage)
    with pytest.raises(IndexError, match="To export to CMYK the image"):
        converter.add_cmyk_data(geoimage, grid2d_gray, "bidon")

    image = Image.fromarray(
        np.random.randint(0, 255, (128, 128, 4)).astype("uint8"), "CMYK"
    )

    geoimage.image = image

    geoimage.to_grid2d(new_name="CMYK", transform="CMYK")
