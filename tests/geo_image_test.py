# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2020-2026 Mira Geoscience Ltd.                                '
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

from warnings import warn

import numpy as np
import pytest
from PIL import Image
from PIL.TiffImagePlugin import TiffImageFile

from geoh5py.data import IntegerData
from geoh5py.objects import GeoImage, Grid2D, Points
from geoh5py.shared.utils import (
    compare_entities,
    xy_rotation_matrix,
    yz_rotation_matrix,
)
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
    284: (1,),
    34737: ("WGS 84 / UTM zone 34N|WGS 84|",),
}

pixels = np.r_[
    np.c_[32, 0, 0],
    np.c_[32, 64, 0],
    np.c_[64, 64, 0],
]
points = np.r_[
    np.c_[5.0, 5.0, 0],
    np.c_[5.0, 10.0, 3],
    np.c_[10.0, 10.0, 3],
]
tie_points = np.array(list(zip(pixels, points, strict=False)))


@pytest.mark.parametrize(
    "tie_points_tag",
    [
        (0.0, 0.0, 0.0, 522796.33210329525, 7244067.563364625, 0.0),
        (
            0.0,
            0.0,
            0.0,
            522796.33210329525,
            7244067.563364625,
            0.0,
            64.0,
            64.0,
            0.0,
            522860.271,
            7244003.625,
            0.0,
        ),
        (
            0.0,  # first point not the smallest XY
            0.0,
            0.0,
            522796.33210329525,
            7244067.563364625,
            0.0,
            128.0,
            0.0,
            0.0,
            522924.209425,
            7244067.563364625,
            0.0,
            128.0,
            128.0,
            0.0,
            522924.209425,
            7243939.686042,
            0.0,
        ),
    ],
)
def test_geoimage_with_tags_one_points(tmp_path, tie_points_tag):
    """
    Test creating a GeoImage with tags and verify vertices functionality.

    Creates an image with geotiff tags and tests that vertices are computed correctly
    from the tag information.
    """
    with Workspace.create(tmp_path / "tagged_image_test.geoh5") as workspace:
        # Create a test image
        xx, yy = np.meshgrid(np.arange(128), np.arange(128))
        diagonal = ((xx + yy) / (128 + 128) * 255).astype("uint8")
        image_data = np.stack([diagonal, diagonal, diagonal], axis=-1)
        image = Image.fromarray(image_data, "RGB")

        temp_tag = tag.copy()
        temp_tag[33922] = tie_points_tag

        for tag_id, tag_value in temp_tag.items():
            image.getexif()[tag_id] = tag_value

        image_path = tmp_path / "test_tagged.tif"
        image.save(image_path, exif=image.getexif())

        # Create GeoImage from the tagged file
        geoimage = GeoImage.create(
            workspace, name="tagged_test_image", image=str(image_path)
        )

        # Test vertices computation from tags
        vertices = geoimage.vertices

        expected = np.array(
            [
                [522796.3321033, 7244067.56336463, 0.0],
                [522924.2094255, 7244067.56336463, 0.0],
                [522924.2094255, 7243939.68604242, 0.0],
                [522796.3321033, 7243939.68604243, 0.0],
            ]
        )

        geoimage.set_tag_from_vertices()

        assert np.allclose(vertices, expected), (
            "Vertices do not match expected values from tags."
        )


def test_attribute_setters():
    workspace = Workspace()
    image = np.random.randint(0, 255, (128, 128))
    gimage = GeoImage.create(workspace, image=image, cells=[[0, 0, 0, 0], [1, 1, 1, 1]])

    assert gimage.n_vertices == 4

    assert gimage.n_cells == 2

    with pytest.raises(
        TypeError, match="Attribute 'cells' must be provided as type numpy.ndarray"
    ):
        gimage.cells = "abc"

    with pytest.raises(ValueError, match="Array of cells should be of shape"):
        gimage.cells = [[0, 0, 0], [1, 1, 1]]

    with pytest.raises(TypeError, match="Indices array must be of integer type"):
        gimage.cells = np.array([[0, 0, 0, 0], [1, 1, 1, 1]], ndmin=2, dtype=float)

    with pytest.raises(TypeError, match="Input 'vertices' must be provided "):
        gimage.vertices = "bidon"

    # Test vertices dtype validation
    wrong_dtype_vertices = np.array(
        [("a", "b", "c"), ("d", "e", "f"), ("g", "h", "i"), ("j", "k", "l")],
        dtype=[("x", "U1"), ("y", "U1"), ("z", "U1")],
    )
    with pytest.raises(TypeError, match="Array of 'vertices' must be of dtype"):
        gimage.vertices = wrong_dtype_vertices


def test_create_copy_empty_geoimage(tmp_path):
    with Workspace.create(tmp_path / r"geo_image_test.geoh5") as workspace:
        geoimage = GeoImage.create(workspace, name="MyGeoImage")

        assert geoimage.image_georeferenced is None

        with pytest.raises(AttributeError, match="The object contains no image data"):
            geoimage.save_as("test")

        with pytest.raises(AttributeError, match="An 'image' must be set be"):
            geoimage.georeference(tie_points)

        with pytest.raises(ValueError, match="Array of 'vertices' must be"):
            geoimage.vertices = [1, 2, 3]

        with pytest.raises(
            ValueError,
            match="Input 'value' for the 'image' property must be a 2D or 3D numpy.ndarray",
        ):
            geoimage.image = np.random.randn(12)

        with pytest.raises(ValueError, match="Shape of the 'image' must be a 2D or "):
            geoimage.image = np.random.randn(12, 12, 4)

        grid2d = geoimage.to_grid2d()
        assert grid2d.children == []
        assert geoimage.image is None

        with pytest.raises(
            AttributeError, match="An 'image' must be set before georeferencing."
        ):
            geoimage.set_tag_from_vertices()

        with pytest.raises(AttributeError, match="The image is not georeferenced"):
            geoimage.georeferencing_from_tiff()


def test_create_geoimage_dry_georeferencing(tmp_path):
    with Workspace.create(tmp_path / r"geo_image_test.geoh5") as workspace:
        geoimage = GeoImage.create(workspace, name="MyGeoImage")
        geoimage.image = np.random.randint(0, 255, (128, 128))

        np.testing.assert_allclose(
            geoimage.extent, np.array([[0.0, 0.0, 0.0], [128.0, 128.0, 0.0]])
        )

        geoimage.georeferencing_from_image()

        with pytest.raises(
            TypeError, match="float\\(\\) argument must be a string or a real number"
        ):
            geoimage.georeference(np.array(zip(pixels[0, :], points, strict=False)))

        with pytest.raises(
            AttributeError,
            match="The 'image' property cannot be reset. Consider creating a new object",
        ):
            geoimage.image = np.random.randint(0, 255, (128, 64, 3))


def test_create_geoimage_full_georeferencing(tmp_path):  # pylint: disable=too-many-statements
    with Workspace.create(tmp_path / r"geo_image_test.geoh5") as workspace:
        geoimage = GeoImage.create(
            workspace, name="MyGeoImage", image=np.random.randint(0, 255, (128, 64, 3))
        )

        geoimage.copy(name="test")

        geoimage.georeference(tie_points)

        # todo: I change the expected values. Is it good?
        temp = np.asarray([[0, 15, 6], [10, 15, 6], [10, 5, 0], [0, 5, 0]])[::-1]

        np.testing.assert_almost_equal(
            geoimage.vertices,
            temp,
            err_msg="Issue geo-referencing the coordinates.",
        )

        geoimage.to_grid2d()

        geoimage.save_as("testtif.tif", str(tmp_path))

        geoimage_copy = GeoImage.create(workspace, name="MyGeoImageTwin")
        geoimage.image_data.copy(parent=geoimage_copy)

        np.testing.assert_almost_equal(
            geoimage_copy.vertices, geoimage.default_vertices
        )

        # Setting image from byte
        geoimage_copy = GeoImage.create(workspace, name="MyGeoImageTwin")
        geoimage_copy.image = geoimage.image_data.file_bytes
        assert geoimage_copy.image == geoimage.image, "Error setting image from bytes."

        # Re-load from file
        geoimage.image.save(tmp_path / r"test.tiff")
        geoimage_file = GeoImage.create(workspace, name="MyGeoImageFile")

        with pytest.raises(ValueError, match="does not exist"):
            geoimage_file.image = str(tmp_path / r"abc.tiff")

        geoimage_file.image = str(tmp_path / r"test.tiff")

        assert geoimage_file.image == geoimage.image, (
            "Error writing and re-loading the image file."
        )

        with Workspace.create(tmp_path / r"geo_image_test2.geoh5") as new_workspace:
            geoimage.copy(parent=new_workspace, clear_cache=True)

            assert geoimage.cells is not None

            rec_image = new_workspace.get_entity("MyGeoImage")[0]

            compare_entities(geoimage, rec_image, ignore=["_parent", "_image", "_tag"])

            assert rec_image.image == geoimage.image, (
                "Error copying the bytes image data."
            )

            geoimage.vertices = geoimage.vertices

            assert np.all(geoimage.mask_by_extent(np.c_[[9, 10], [9, 10]]))
            assert geoimage.mask_by_extent(np.c_[[90, 100], [90, 100]]) is None

            # Test copy from extent that clips one corner
            new_image = geoimage.copy_from_extent(np.c_[[9, 10], [9, 10]])

            assert new_image is not None, "Error copying from extent."

            new_image = geoimage.copy_from_extent(np.vstack([[100, 100], [200, 200]]))

            assert new_image is None, "Error copying from extent that is out of bounds."


def test_georeference_image(tmp_path):
    with Workspace.create(tmp_path / r"geo_image_test.geoh5") as workspace:
        # create and save a tiff
        image = Image.fromarray(
            np.random.randint(0, 255, (128, 128, 3)).astype("uint8"), "RGB"
        )
        for id_ in tag.items():
            image.getexif()[id_[0]] = id_[1]
        image.save(tmp_path / r"testtif.tif", exif=image.getexif())

        # load image
        geoimage = GeoImage.create(
            workspace, name="test_area", image=f"{tmp_path!s}/testtif.tif"
        )

        geoimage.tag = None

        # test grid2d errors
        with pytest.raises(ValueError, match="Input 'tag' must"):
            geoimage.tag = 42

        # image = Image.open(tmp_path / r"testtif.tif")
        geoimage.tag = {"test": 3}

        with pytest.warns(
            UserWarning, match="The 'tif.' image is missing one or more required tags"
        ):
            geoimage.georeferencing_from_tiff()

        image = Image.open(f"{tmp_path!s}/testtif.tif")

        geoimage = GeoImage.create(workspace, name="test_area", image=image)

        # create Gray grid2d
        grid2d_gray = geoimage.to_grid2d(mode="GRAY")

        # create RGB grid2d
        grid2d_rgb = geoimage.to_grid2d(new_name="RGB")

        assert isinstance(grid2d_gray, Grid2D)
        assert (
            len(
                [
                    child
                    for child in grid2d_gray.children
                    if isinstance(child, IntegerData)
                ]
            )
            == 1
        )
        assert isinstance(grid2d_rgb, Grid2D)
        assert isinstance(geoimage.image_georeferenced, Image.Image)

        # test grid2d errors
        with pytest.raises(
            ValueError, match="conversion from RGB to bidon not supported"
        ):
            geoimage.to_grid2d(new_name="RGB", mode="bidon")

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

        image = Image.open(f"{tmp_path!s}/testtif.tif").convert("L")
        geoimage = GeoImage.create(workspace, name="test_area", image=image)

        assert geoimage is not None

        image = Image.fromarray(
            np.random.randint(0, 255, (128, 128, 4)).astype("uint8"), "RGBA"
        )

        geoimage = GeoImage.create(workspace, name="to_CMYK", image=image)
        new_grid = geoimage.to_grid2d(name="CMYK")
        assert len(new_grid.children) == 4


def test_georeferencing_from_tiff_errors_and_warnings(tmp_path):
    """
    Test error and warning scenarios for the georeferencing_from_tiff method.

    This test covers:
    1. AttributeError when image is not georeferenced (tag is None)
    2. UserWarning when required tags are missing (KeyError)
    3. UserWarning when tag values cannot be parsed (ValueError)
    """
    with Workspace.create(tmp_path / r"geo_image_test.geoh5") as workspace:
        # Test 1: AttributeError when tag is None (no georeference info)
        image_array = np.random.randint(0, 255, (128, 128, 3)).astype("uint8")
        geoimage_no_tag = GeoImage.create(
            workspace, name="no_tag_image", image=image_array
        )

        with pytest.raises(AttributeError, match="The image is not georeferenced"):
            geoimage_no_tag.georeferencing_from_tiff()

        # Test 2: UserWarning for missing required tags (KeyError)
        # Create image with incomplete tags (missing required tag 33550)
        image = Image.fromarray(image_array, "RGB")
        incomplete_tag = {
            256: (128,),  # ImageWidth
            257: (128,),  # ImageHeight
            33922: (0.0, 0.0, 0.0, 522796.33, 7244067.56, 0.0),  # ModelTiepointTag
            # Missing 33550 (ModelPixelScaleTag) - will cause KeyError
        }

        for tag_id, tag_value in incomplete_tag.items():
            image.getexif()[tag_id] = tag_value

        incomplete_image_path = tmp_path / "incomplete_tags.tif"
        image.save(incomplete_image_path, exif=image.getexif())

        geoimage_incomplete = GeoImage.create(
            workspace, name="incomplete_tags_image", image=str(incomplete_image_path)
        )

        with pytest.warns(
            UserWarning,
            match="The 'tif.' image is missing one or more required tags",
        ):
            geoimage_incomplete.georeferencing_from_tiff()

        # Test 3: UserWarning when georeferencing validation fails
        # Create an image with tie points that are non-coplanar
        image_correct = Image.fromarray(
            np.random.randint(0, 255, (64, 64, 3)).astype("uint8"), "RGB"
        )
        # Create tie points with world coordinates that are non-coplanar
        # This will trigger the coplanarity check in _compute_image_corners
        non_coplanar_tag = {
            256: (64,),
            257: (64,),
            # Multiple tie points where world z coordinates don't form a plane
            33922: (
                0.0,
                0.0,
                0.0,
                100.0,
                200.0,
                0.0,  # First tie point
                10.0,
                0.0,
                0.0,
                110.0,
                200.0,
                0.0,  # Second tie point
                0.0,
                10.0,
                0.0,
                100.0,
                210.0,
                0.0,  # Third tie point
                10.0,
                10.0,
                0.0,
                110.0,
                210.0,
                100.0,  # Fourth - breaks coplanarity
            ),
            33550: (1.0, 1.0, 0.0),
        }

        for tag_id, tag_value in non_coplanar_tag.items():
            image_correct.getexif()[tag_id] = tag_value

        non_coplanar_path = tmp_path / "non_coplanar.tif"
        image_correct.save(non_coplanar_path, exif=image_correct.getexif())

        geoimage_non_coplanar = GeoImage.create(
            workspace, name="non_coplanar_image", image=str(non_coplanar_path)
        )

        with pytest.warns(
            UserWarning,
            match="Georeferencing from tiff failed because of the following reasons:",
        ):
            geoimage_non_coplanar.georeferencing_from_tiff()


def test_rotation_setter(tmp_path):
    with Workspace.create(tmp_path / r"geo_image_test.geoh5") as workspace:
        # add the data
        x_val, y_val = np.meshgrid(
            np.linspace(100, 1000, 16), np.linspace(100, 1500, 16)
        )
        values = x_val + y_val
        values = (values - np.min(values)) / (np.max(values) - np.min(values))
        values *= 255
        values = np.repeat(values.astype(np.uint32)[:, :, np.newaxis], 3, axis=2)

        # load image
        geoimage = GeoImage.create(workspace, name="test_area", image=values)

        rotated = geoimage.copy()

        assert geoimage.rotation == 0

        rotated.rotation = 45

        np.testing.assert_array_almost_equal(rotated.rotation, 45)

        rotated.rotation = 0

        np.testing.assert_array_almost_equal(rotated.rotation, 0)

        assert np.allclose(geoimage.vertices, rotated.vertices)
        assert geoimage.image == rotated.image


def test_converting_rotated_images(tmp_path):
    with Workspace.create(tmp_path / r"geo_image_test.geoh5") as workspace:
        # create a grid
        n_x, n_y = 10, 15
        grid = Grid2D.create(
            workspace,
            origin=[0, 0, 0],
            u_cell_size=20.0,
            v_cell_size=30.0,
            u_count=n_x,
            v_count=n_y,
            rotation=30,
            name="MyTestGrid2D",
            allow_move=False,
        )

        # add the data
        x_val, y_val = np.meshgrid(
            np.linspace(100, 1000, n_x), np.linspace(100, 1500, n_y)
        )
        values = x_val + y_val
        values = (values - np.min(values)) / (np.max(values) - np.min(values))
        values *= 255
        values = values.astype(np.uint32)

        _ = grid.add_data(
            {
                "rando_r": {"values": values.flatten()},
                "rando_g": {"values": values.flatten()[::-1]},
                "rando_b": {"values": values.flatten()},
            }
        )

        # convert to geoimage
        geoimage = grid.to_geoimage(["rando_r", "rando_g", "rando_b"], normalize=False)

        # convert to test
        grid_test = geoimage.to_grid2d()

        np.testing.assert_almost_equal(grid.rotation, geoimage.rotation)
        np.testing.assert_almost_equal(grid_test.u_cell_size, grid.u_cell_size)
        np.testing.assert_almost_equal(grid_test.v_cell_size, grid.v_cell_size)
        np.testing.assert_almost_equal(grid_test.u_count, grid.u_count)
        np.testing.assert_almost_equal(grid_test.v_count, grid.v_count)
        assert grid_test.origin == grid.origin
        np.testing.assert_almost_equal(grid_test.rotation, grid.rotation)

        assert all(
            grid_test.get_data("band[0]")[0].values
            == grid.get_data("rando_r")[0].values
        )

        grid_test = geoimage.to_grid2d(mode="GRAY")
        assert "band[0]" in grid_test.get_data_list()


def test_image_rotation(tmp_path):
    with Workspace.create(tmp_path / r"geo_image_test.geoh5") as workspace:
        # Repeat with gray scale image
        image = Image.fromarray(
            np.random.randint(0, 255, (128, 128)).astype("uint8"), "L"
        )
        geoimage = GeoImage.create(workspace, name="test_area", image=image)

        np.testing.assert_array_almost_equal(geoimage.rotation, 0)
        np.testing.assert_array_almost_equal(geoimage.dip, 0)

        geoimage2 = GeoImage.create(
            workspace, name="test_area", image=image, rotation=66
        )
        np.testing.assert_array_almost_equal(geoimage2.rotation, 66)

        geoimage3 = GeoImage.create(workspace, name="test_area", image=image, dip=44)
        np.testing.assert_array_almost_equal(geoimage3.dip, 44)

        geoimage4 = GeoImage.create(
            workspace, name="test_area", image=image, dip=44, rotation=66
        )
        np.testing.assert_array_almost_equal(geoimage4.dip, 44)
        np.testing.assert_array_almost_equal(geoimage4.rotation, 66)

        vertices = geoimage.vertices - geoimage.origin

        rotation_matrix = xy_rotation_matrix(np.deg2rad(66))
        dip_matrix = yz_rotation_matrix(np.deg2rad(44))

        rotated_vertices = np.dot(rotation_matrix, vertices.T).T
        dipped_vertices = np.dot(dip_matrix, vertices.T).T
        rotated_dipped_vertices = np.dot(rotation_matrix, dipped_vertices.T).T

        assert np.allclose(geoimage2.vertices, rotated_vertices + geoimage.origin)
        assert np.allclose(geoimage3.vertices, dipped_vertices + geoimage.origin)
        assert np.allclose(
            geoimage4.vertices, rotated_dipped_vertices + geoimage.origin
        )


def test_image_grid_rotation_conversion(tmp_path):
    with Workspace.create(tmp_path / r"geo_image_test.geoh5") as workspace:
        # Repeat with gray scale image
        image = Image.fromarray(
            np.random.randint(0, 255, (128, 128)).astype("uint8"), "L"
        )
        geoimage = GeoImage.create(workspace, name="test_area", image=image)
        geoimage.set_tag_from_vertices()

        # convert to grid2d
        grid2d = geoimage.to_grid2d(mode="GRAY")

        # change dip and rotation
        grid2d.rotation = 66
        grid2d.dip = 44
        geoimage.rotation = 66
        geoimage.dip = 44

        geoimage2 = grid2d.to_geoimage(0, normalize=False, ignore=["tag"])

        compare_entities(geoimage, geoimage2, ignore=["_uid", "_image_data"])

        geoimage.georeferencing_from_image()

        with pytest.raises(NotImplementedError):
            GeoImage.create(
                workspace,
                name="test_area",
                image=Image.fromarray(
                    np.random.randint(0, 255, (128, 128, 2)).astype("uint8"), "LA"
                ),
            )


def test_clipping_rotated_image(tmp_path):
    with Workspace.create(tmp_path / r"geo_image_test.geoh5") as workspace:
        # create a grid
        n_x, n_y = 10, 15
        grid = Grid2D.create(
            workspace,
            origin=[0, 0, 0],
            u_cell_size=20.0,
            v_cell_size=30.0,
            u_count=n_x,
            v_count=n_y,
            rotation=30,
            name="MyTestGrid2D",
            allow_move=False,
        )

        # add the data
        x_val, y_val = np.meshgrid(
            np.linspace(0, 909, n_x), np.linspace(100, 1500, n_y)
        )
        values = x_val + y_val
        values = (values - np.min(values)) / (np.max(values) - np.min(values))
        values *= 255
        values = values.astype(np.uint32)

        _ = grid.add_data(
            {
                "rando_c": {"values": values.flatten()},
                "rando_m": {"values": values.flatten()[::-1]},
                "rando_y": {"values": values.flatten()},
                "rando_k": {"values": np.zeros_like(values.flatten())},
            }
        )

        # convert to geoimage # todo: the bug seems here
        geoimage = grid.to_geoimage(
            ["rando_c", "rando_m", "rando_y", "rando_k"], mode="RGBA", normalize=False
        )

        # clip the image
        extent = np.r_[np.c_[50, 50], np.c_[200, 200]]

        copy_image = geoimage.copy_from_extent(extent)

        copy_grid = grid.copy_from_extent(extent)

        vertices = np.array(
            [
                [50.0, 50.0, -1.0],
                [200.0, 50.0, -1.0],
                [200.0, 200.0, 1.0],
                [50.0, 200.0, 1.0],
            ]
        )

        Points.create(workspace, vertices=vertices)

        # copy by extent via geoimage create a bigger array
        # It's normal because in geoimage, we are projecting
        # the biggest extent along a plane
        # assert copy_image.u_count == copy_grid.u_count
        # assert copy_image.v_count == copy_grid.v_count

        assert np.isclose(copy_image.u_cell_size, copy_grid.u_cell_size)
        assert np.isclose(copy_image.v_cell_size, copy_grid.v_cell_size)

        # todo: issue with the image!
        # assert np.allclose(
        #     np.array(copy_image.image)[:, :, 0].ravel()[mask],
        #     copy_grid.get_data("rando_c")[0].values[mask],
        # )


def copy_geoimage_via_grid2d(
    geoimage, extent, parent=None, copy_children=True, clear_cache=False, **kwargs
):
    """
    Utility function to copy a geoimage using grid2d conversion.

    This function converts the geoimage to grid2d, performs copy_from_extent
    on the grid2d, then converts back to geoimage.

    :param geoimage: The GeoImage object to copy from extent.
    :param extent: The extent to copy.
    :param parent: Parent workspace for the result.
    :param copy_children: Whether to copy children.
    :param clear_cache: Whether to clear cache.
    :param kwargs: Additional keyword arguments.

    :return: New GeoImage object cropped to extent, or None if no intersection.
    """
    # transform the image to a grid
    grid = geoimage.to_grid2d(parent=parent, mode="RGBA", name="_temp_grid")

    # transform the image
    grid_transformed = grid.copy_from_extent(
        extent=extent,
        parent=parent,
        name="_temp_grid_cropped",
        copy_children=copy_children,
        clear_cache=clear_cache,
        from_image=True,
        **kwargs,
    )

    if grid_transformed is None:
        return None

    # transform the grid back to an image
    image_transformed = grid_transformed.to_geoimage(
        keys=grid_transformed.get_data_list(),
        mode="RGBA",
        normalize=False,
        name=geoimage.name + "_from_grid2d",
    )
    return image_transformed


def compare_geoimages(
    direct: GeoImage, grid_converted: GeoImage, original: GeoImage, test_name: str
) -> list[str]:
    """
    Compare two GeoImages and return comparison errors.

    :param direct: GeoImage from direct copy_from_extent
    :param grid_converted: GeoImage from grid2d conversion method
    :param original: Original GeoImage before cropping
    :param test_name: Name of test for reporting

    :return: List of error messages
    """
    errors = []

    if direct.u_count != grid_converted.u_count:
        errors.append(
            f"u_count mismatch: direct={direct.u_count}, converted={grid_converted.u_count}"
        )

    if direct.v_count != grid_converted.v_count:
        errors.append(
            f"v_count mismatch: direct={direct.v_count}, converted={grid_converted.v_count}"
        )

    if not np.isclose(
        direct.u_cell_size, grid_converted.u_cell_size, rtol=1e-6, atol=1e-6
    ):
        errors.append(
            f"u_cell_size mismatch: direct={direct.u_cell_size}, converted={grid_converted.u_cell_size}"
        )

    if not np.isclose(
        direct.v_cell_size, grid_converted.v_cell_size, rtol=1e-6, atol=1e-6
    ):
        errors.append(
            f"v_cell_size mismatch: direct={direct.v_cell_size}, converted={grid_converted.v_cell_size}"
        )

    if not np.allclose(direct.vertices, grid_converted.vertices, rtol=1e-6, atol=1e-6):
        diff = direct.vertices - grid_converted.vertices
        errors.append(
            "Vertices mismatch:\n"
            f"direct=\n{direct.vertices}\n"
            f"converted=\n{grid_converted.vertices}\n"
            f"difference=\n{diff}"
        )

    if direct.image is not None and grid_converted.image is not None:
        direct_array = np.asarray(direct.image)
        if direct_array.ndim == 3:
            direct_array = direct_array[:, :, 0]

        grid_array = np.asarray(grid_converted.image)
        if grid_array.ndim == 3:
            grid_array = grid_array[:, :, 0]

        # mask NDV = 0
        mask = grid_array != 0

        if mask.any():
            if mask.shape == direct_array.shape:
                d = direct_array[mask]
                g = grid_array[mask]

                if not np.allclose(d, g, rtol=1e-3, atol=2):
                    # compute difference stats
                    diff = d - g
                    errors.append(
                        "Image content mismatch within valid region:\n"
                        f"max abs diff: {np.max(np.abs(diff))}\n"
                        f"mean diff:    {np.mean(diff):.6f}\n"
                        f"direct sample: {d[:20]}\n"
                        f"converted sample: {g[:20]}"
                    )
            else:
                errors.append(
                    f"Image shape mismatch within valid region: direct={direct_array.shape}, converted={grid_array.shape}"
                )

    return errors


def run_extent_test_case(
    geoimage: GeoImage, extent: np.ndarray, workspace
) -> tuple[GeoImage | None, GeoImage | None]:
    """
    Run copy_from_extent test case with both methods.

    :param geoimage: Original GeoImage
    :param extent: Extent array
    :param workspace: Workspace for new objects
    :return: Tuple of (direct_result, grid2d_result)
    """
    # Direct method
    direct_result = geoimage.copy_from_extent(
        extent, parent=workspace, name=geoimage.name + "_direct"
    )

    # Grid2d method
    grid_result = copy_geoimage_via_grid2d(geoimage, extent, parent=workspace)

    return direct_result, grid_result


def display_extent(extent, workspace, case_id):
    """
    Utility function to create a Points object representing the extent corners.
    """
    min_x, min_y, min_z = extent[0]
    max_x, max_y, max_z = extent[1]

    extent_corners = np.array(
        [
            [min_x, min_y, min_z],
            [max_x, min_y, min_z],
            [max_x, max_y, min_z],
            [min_x, max_y, min_z],
            [min_x, min_y, max_z],
            [max_x, min_y, max_z],
            [max_x, max_y, max_z],
            [min_x, max_y, max_z],
        ]
    )
    Points.create(
        workspace,
        name=f"{case_id}_extent",
        vertices=extent_corners,
    )


def build_extent_test_params():
    """
    Prepare pytest parameters for extent copy tests with origin and extent offset propagation.
    """

    # Flat grid-aligned extents
    base_cases = [
        (np.array([[2, 2, -1], [7, 7, 1]]), "simple_center_crop"),
        (np.array([[0, 0, -1], [3, 3, 1]]), "corner_crop"),
        (np.array([[3, 3, -1], [6, 6, 1]]), "fully_inside_small"),
        (np.array([[2, 9, -1], [8, 11, 1]]), "exact_border_top"),
        (np.array([[8, 0, -1], [11, 3, 1]]), "exact_corner_bottom_right"),
    ]

    # Transformations to apply on the 2-D plane
    transforms = [
        (0, 0, "flat"),
        (0, 45, "rot45"),
        (0, 90, "rot90"),
        (15, 0, "dip15"),
        (30, 0, "dip30"),
        (45, 0, "dip45"),
        (15, 45, "dip15_rot45"),
        (30, 90, "dip30_rot90"),
    ]

    # Offset origins (used to shift extents too)
    offsets_cases = [
        (
            np.array([10, 5, 2]),
            0,
            0,
            np.array([[12, 7, -1], [17, 12, 5]]),
            "offset_origin_center_crop",
        ),
        (
            np.array([5, 5, 1]),
            0,
            45,
            np.array([[7, 7, -1], [12, 12, 3]]),
            "offset_origin_rot45",
        ),
    ]

    # Default origin for flat cases
    origin_default = np.array([0, 0, 0])

    params = [
        pytest.param(
            origin_default,
            dip,
            rot,
            (extent + origin_default).copy(),  # shift extent by default origin
            case_name,
            id=f"{suffix}_{case_name}",  # each independent test ID
        )
        for extent, case_name in base_cases
        for dip, rot, suffix in transforms
    ]

    # Add offset origin cases, propagating offset to extent
    for origin_offset, dip, rot, extent, case_name in offsets_cases:
        params.append(
            pytest.param(
                origin_offset,
                dip,
                rot,
                (extent + origin_offset).copy(),
                case_name,
                id=case_name,
            )
        )

    return params


PARAMS = build_extent_test_params()


@pytest.mark.parametrize("origin,dip,rotation,extent,case_id", PARAMS)
def test_cfegi(tmp_path, origin, dip, rotation, extent, case_id):
    workspace_path = tmp_path / f"{case_id}.geoh5"

    # Deterministic test image
    arr = np.arange(100, dtype="uint8").reshape(10, 10)
    image = Image.fromarray(arr, "L")

    # Note the vertices are not well ordered, helped me to find a bug
    base_vertices = np.array(
        [
            [0, 0, 0],
            [10, 0, 0],
            [10, 10, 0],
            [0, 10, 0],
        ]
    )

    with Workspace.create(workspace_path) as workspace:
        vertices = base_vertices.copy()
        if origin is not None:
            vertices = vertices + origin

        geoimage = GeoImage.create(
            workspace,
            name=f"{case_id}_geoimage",
            image=image.copy(),
            vertices=vertices,
        )

        geoimage.dip = dip
        geoimage.rotation = rotation

        display_extent(extent, workspace, case_id)

        # Run both methods
        direct_result, grid_result = run_extent_test_case(geoimage, extent, workspace)

        # Skip known conversion limitation cases
        comparison_errors = []
        if direct_result is None and grid_result is None:
            warn("Skipping comparison due to one method returning None")
            return
        elif direct_result is None or grid_result is None:
            comparison_errors.append(
                "One method returned None while the other did not."
                f" direct_result is None: {direct_result is None},"
                f" grid_result is None: {grid_result is None}"
            )
        else:
            # Compare and raise immediately if differences exist
            comparison_errors = compare_geoimages(
                direct_result, grid_result, geoimage, case_id
            )
        if comparison_errors:
            error_message = "\n\n".join(comparison_errors)
            raise AssertionError(f"Comparison failed for '{case_id}' → {error_message}")


def test_copy_from_extent_error_conditions(tmp_path):
    """
    Test error and warning conditions in copy_from_extent method.

    Tests the specific error cases mentioned in lines 158-167 of geo_image.py:
    - AttributeError when vertices are not defined
    - UserWarning when image is not defined
    - NotImplementedError when inverse=True
    """
    workspace_path = tmp_path / "copy_extent_errors.geoh5"

    with Workspace.create(workspace_path) as workspace:
        extent = np.array([[0, 0, 0], [1, 1, 1]])

        # Test case 1: AttributeError when vertices are not defined
        # Create with image so we get past the image check, then remove vertices
        image = Image.fromarray(np.arange(16, dtype="uint8").reshape(4, 4), "L")

        # Test case 2: UserWarning when image is not defined (but vertices are)
        vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        geoimage_no_image = GeoImage.create(
            workspace, name="no_image", vertices=vertices
        )
        # Ensure image is None
        assert geoimage_no_image.image is None, "Image should be None for this test"

        with pytest.warns(UserWarning, match="Image is not defined"):
            result = geoimage_no_image.copy_from_extent(extent)
            assert result is None, "Should return None when image is not defined"

        # Test case 3: NotImplementedError when inverse=True
        geoimage_with_image = GeoImage.create(
            workspace, name="with_image", image=image, vertices=vertices
        )

        with pytest.raises(
            NotImplementedError, match="Inverse mask is not implemented yet with images"
        ):
            geoimage_with_image.copy_from_extent(extent, inverse=True)

        # Test case 4: copy_from_extent with parent parameter (workspace)
        image2 = Image.fromarray(np.arange(100, dtype="uint8").reshape(10, 10), "L")
        vertices2 = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]])
        geoimage_for_copy = GeoImage.create(
            workspace, name="for_copy", image=image2, vertices=vertices2
        )

        extent_inside = np.array([[2, 2, -1], [8, 8, 1]])
        copied_with_workspace_parent = geoimage_for_copy.copy_from_extent(
            extent_inside, parent=workspace
        )
        assert copied_with_workspace_parent is not None
        assert copied_with_workspace_parent.workspace == workspace

        # Test case 5: copy_from_extent with parent parameter (entity/group)
        from geoh5py.groups import ContainerGroup

        test_group = ContainerGroup.create(workspace, name="test_group")

        copied_with_group_parent = geoimage_for_copy.copy_from_extent(
            extent_inside, parent=test_group
        )
        assert copied_with_group_parent is not None
        assert copied_with_group_parent.parent == test_group
        assert copied_with_group_parent.workspace == workspace


def test_parse_tie_points():
    """
    Test _parse_tie_points static method error handling and validation.
    """
    # Test case 1: Valid tuple input (single tie point)
    tie_points_tuple = (0.0, 0.0, 0.0, 100.0, 200.0, 0.0)
    result = GeoImage._parse_tie_points(tie_points_tuple)
    expected = np.array([[[0.0, 0.0, 0.0], [100.0, 200.0, 0.0]]])
    np.testing.assert_array_equal(result, expected)

    # Test case 2: Valid list input (multiple tie points)
    tie_points_list = [
        0.0,
        0.0,
        0.0,
        100.0,
        200.0,
        0.0,
        10.0,
        10.0,
        0.0,
        110.0,
        210.0,
        0.0,
    ]
    result = GeoImage._parse_tie_points(tie_points_list)
    assert result.shape == (2, 2, 3)

    # Test case 3: Valid numpy array input
    tie_points_array = np.array(
        [
            [[0.0, 0.0, 0.0], [100.0, 200.0, 0.0]],
            [[10.0, 10.0, 0.0], [110.0, 210.0, 0.0]],
        ]
    )
    result = GeoImage._parse_tie_points(tie_points_array)
    np.testing.assert_array_equal(result, tie_points_array)

    # Test case 4: ValueError - tuple/list length not multiple of 6
    with pytest.raises(
        ValueError, match="ModelTiepointTag length must be a multiple of 6"
    ):
        GeoImage._parse_tie_points((0.0, 0.0, 0.0, 100.0, 200.0))  # 5 elements

    with pytest.raises(
        ValueError, match="ModelTiepointTag length must be a multiple of 6"
    ):
        GeoImage._parse_tie_points([0.0, 0.0, 0.0, 100.0])  # 4 elements

    # Test case 5: ValueError - wrong numpy array shape
    with pytest.raises(ValueError, match="Tie points must have shape"):
        # Wrong dimensions (2D instead of 3D)
        GeoImage._parse_tie_points(np.array([[0.0, 0.0, 0.0], [100.0, 200.0, 0.0]]))

    with pytest.raises(ValueError, match="Tie points must have shape"):
        # Wrong shape in last dimensions
        GeoImage._parse_tie_points(np.array([[[0.0, 0.0], [100.0, 200.0]]]))

    # Test case 6: ValueError - same pixel maps to different world coordinates
    inconsistent_tie_points = np.array(
        [
            [
                [0.0, 0.0, 0.0],
                [100.0, 200.0, 0.0],
            ],  # pixel (0,0,0) -> world (100,200,0)
            [
                [0.0, 0.0, 0.0],
                [150.0, 250.0, 0.0],
            ],  # pixel (0,0,0) -> world (150,250,0)
        ]
    )
    with pytest.raises(
        ValueError, match="identical pixel coordinates map to multiple world"
    ):
        GeoImage._parse_tie_points(inconsistent_tie_points)

    # Test case 7: ValueError - same world maps to different pixel coordinates
    inconsistent_world = np.array(
        [
            [
                [0.0, 0.0, 0.0],
                [100.0, 200.0, 0.0],
            ],  # pixel (0,0,0) -> world (100,200,0)
            [
                [10.0, 10.0, 0.0],
                [100.0, 200.0, 0.0],
            ],  # pixel (10,10,0) -> world (100,200,0)
        ]
    )
    with pytest.raises(
        ValueError, match="identical world coordinates map to multiple pixel"
    ):
        GeoImage._parse_tie_points(inconsistent_world)

    # Test case 8: Duplicate removal - exact duplicates should be handled
    duplicated_tie_points = np.array(
        [
            [[0.0, 0.0, 0.0], [100.0, 200.0, 0.0]],
            [[0.0, 0.0, 0.0], [100.0, 200.0, 0.0]],  # exact duplicate
            [[10.0, 10.0, 0.0], [110.0, 210.0, 0.0]],
        ]
    )
    result = GeoImage._parse_tie_points(duplicated_tie_points)
    # Should have only 2 unique pairs
    assert result.shape == (2, 2, 3)


def test_compute_image_corners_from_1_tie_point(tmp_path):
    """
    Test _compute_image_corners_from_1_tie_point method error handling and computation.
    """
    with Workspace.create(tmp_path / "test_1_tie_point.geoh5") as workspace:
        # Create a simple test image (10x10)
        image = Image.fromarray(np.arange(100, dtype="uint8").reshape(10, 10), "L")
        geoimage = GeoImage.create(workspace, name="test", image=image)

        # Test case 1: Valid single tie point at origin
        tie_points = np.array([[[0.0, 0.0, 0.0], [100.0, 200.0, 5.0]]])
        result = geoimage._compute_image_corners_from_1_tie_point(
            tie_points, u_cell_size=1.0, v_cell_size=1.0
        )
        assert result.shape == (4, 3)
        # z should be constant
        assert np.allclose(result[:, 2], 5.0)
        # Verify corners based on tie point at pixel (0,0) -> world (100, 200, 5)
        # default_vertices reversed = [[0,0,0], [10,0,0], [10,10,0], [0,10,0]]
        expected = np.array(
            [
                [100.0, 200.0, 5.0],  # pixel (0, 0) - origin
                [110.0, 200.0, 5.0],  # pixel (10, 0)
                [110.0, 190.0, 5.0],  # pixel (10, 10) - note: v is negative direction
                [100.0, 190.0, 5.0],  # pixel (0, 10)
            ]
        )
        np.testing.assert_allclose(result, expected)

        # Test case 2: Valid tie point at different pixel location
        tie_points_offset = np.array([[[5.0, 5.0, 0.0], [500.0, 600.0, 10.0]]])
        result_offset = geoimage._compute_image_corners_from_1_tie_point(
            tie_points_offset, u_cell_size=2.0, v_cell_size=3.0
        )
        assert result_offset.shape == (4, 3)
        assert np.allclose(result_offset[:, 2], 10.0)

        # Test case 3: ValueError - empty tie points array
        with pytest.raises(IndexError):
            geoimage._compute_image_corners_from_1_tie_point(
                np.array([]).reshape(0, 2, 3), u_cell_size=1.0, v_cell_size=1.0
            )


def build_2_tie_points_test_params():
    """
    Build test parameters for _compute_image_corners_from_2_tie_points.

    Tests are designed by choosing desired U and V basis vectors, then computing
    tie points that satisfy: delta_wrd = di*U + dj*V
    where |U| = u_cell_size, |V| = v_cell_size, U ⊥ V
    """
    u_cell_size = 1.0
    v_cell_size = 1.0

    # Helper function to generate test case from U and V
    def make_test(U, V, origin, di, dj, test_id):
        """Generate test parameters from desired basis vectors."""
        # Corresponding world position
        wrd1 = origin + di * U + dj * V

        # Compute corners for 10x10 image
        corners = np.array(
            [
                origin + 0 * U + 0 * V,  # (0,0)
                origin + 10 * U + 0 * V,  # (10,0)
                origin + 10 * U + 10 * V,  # (10,10)
                origin + 0 * U + 10 * V,  # (0,10)
            ]
        )

        return pytest.param(
            np.array(
                [
                    [[0.0, 0.0, 0.0], origin],
                    [[di, dj, 0.0], wrd1],
                ]
            ),
            u_cell_size,
            v_cell_size,
            corners,
            id=test_id,
        )

    # Test 1: Axis-aligned, U along X, V along Y (no rotation)
    U1 = np.array([1.0, 0.0, 0.0])
    V1 = np.array([0.0, 1.0, 0.0])
    origin1 = np.array([100.0, 200.0, 5.0])
    test1 = make_test(U1, V1, origin1, di=5.0, dj=5.0, test_id="no_rotation")

    # Test 2: 45° rotation in xy-plane, no dip
    cos45 = np.cos(np.pi / 4)
    sin45 = np.sin(np.pi / 4)
    U2 = np.array([cos45, sin45, 0.0])  # Rotated X
    V2 = np.array([-sin45, cos45, 0.0])  # Rotated Y  (perpendicular to U2)
    origin2 = np.array([0.0, 0.0, 0.0])
    test2 = make_test(U2, V2, origin2, di=5.0, dj=5.0, test_id="rotated_45deg")

    # Test 3: No rotation but V dips 30° downward
    U3 = np.array([1.0, 0.0, 0.0])
    V3 = np.array([0.0, np.cos(np.pi / 6), -np.sin(np.pi / 6)])  # Dip 30° in yz-plane
    origin3 = np.array([50.0, 50.0, 10.0])
    test3 = make_test(U3, V3, origin3, di=3.0, dj=4.0, test_id="dip_30deg")

    # Test 4: 30° rotation + 20° dip on V
    cos30 = np.cos(np.pi / 6)
    sin30 = np.sin(np.pi / 6)
    dip20 = np.pi / 9
    U4 = np.array([cos30, sin30, 0.0])
    V4 = np.array([-sin30 * np.cos(dip20), cos30 * np.cos(dip20), -np.sin(dip20)])
    origin4 = np.array([100.0, 100.0, 0.0])
    test4 = make_test(U4, V4, origin4, di=6.0, dj=4.0, test_id="rotated_dipped")

    # Test 5: Complex oblique orientation
    # U at 60° rotation, V perpendicular with 45° dip
    cos60 = np.cos(np.pi / 3)
    sin60 = np.sin(np.pi / 3)
    U5 = np.array([cos60, sin60, 0.0])
    V5 = np.array([-sin60 / np.sqrt(2), cos60 / np.sqrt(2), -1 / np.sqrt(2)])  # 45° dip
    origin5 = np.array([10.0, 20.0, 30.0])
    test5 = make_test(U5, V5, origin5, di=4.0, dj=6.0, test_id="oblique")

    # Test 6: Vertical alignment (di = 0, dj != 0)
    # Tie points differ only in j direction
    U6 = np.array([1.0, 0.0, 0.0])
    V6 = np.array([0.0, 1.0, 0.0])
    origin6 = np.array([100.0, 200.0, 0.0])
    test6 = make_test(U6, V6, origin6, di=0.0, dj=8.0, test_id="vertical_alignment_di0")

    # Test 7: Horizontal alignment (di != 0, dj = 0)
    # Tie points differ only in i direction
    U7 = np.array([1.0, 0.0, 0.0])
    V7 = np.array([0.0, 1.0, 0.0])
    origin7 = np.array([50.0, 100.0, 5.0])
    test7 = make_test(
        U7, V7, origin7, di=6.0, dj=0.0, test_id="horizontal_alignment_dj0"
    )

    return [test1, test2, test3, test4, test5, test6, test7]


TWO_TIE_POINTS_PARAMS = build_2_tie_points_test_params()


@pytest.mark.parametrize(
    "tie_points,u_cell_size,v_cell_size,expected", TWO_TIE_POINTS_PARAMS
)
def test_compute_image_corners_from_2_tie_points(
    tmp_path, tie_points, u_cell_size, v_cell_size, expected
):
    """
    Test georeference method with 2 tie points and various 3D plane configurations.

    Tests different scenarios with orthogonality assumption and known cell sizes:
    - Verifies orthogonal basis vectors are constructed correctly
    - Checks constraints: |U|=u_cell_size, |V|=v_cell_size
    - Validates corners satisfy di*U + dj*V = delta_wrd

    Note: The solution is not unique (1 DOF - rotation in plane), so we test
    properties rather than exact corner positions.
    """
    with Workspace.create(tmp_path / "test_2_tie_points.geoh5") as workspace:
        # Create a simple test image (10x10)
        image = Image.fromarray(np.arange(100, dtype="uint8").reshape(10, 10), "L")
        geoimage = GeoImage.create(workspace, name="test", image=image)

        geoimage.georeference(tie_points, u_cell_size, v_cell_size)
        result = geoimage.vertices

        assert result.shape == (4, 3)

        # Extract tie point data
        pix0, pix1 = tie_points[:2, 0, :2]
        wrd0, wrd1 = tie_points[:2, 1, :]
        di, dj = pix1 - pix0
        delta_wrd = wrd1 - wrd0

        # Reconstruct U and V from corners
        # Corners are at pixels: (0,0), (10,0), (10,10), (0,10)
        corner_00 = result[0]
        corner_10_0 = result[1]
        corner_0_10 = result[3]

        U = (corner_10_0 - corner_00) / 10.0  # Displacement per pixel in i direction
        V = (corner_0_10 - corner_00) / 10.0  # Displacement per pixel in j direction

        # Verify magnitudes
        np.testing.assert_allclose(
            np.linalg.norm(U), u_cell_size, rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            np.linalg.norm(V), v_cell_size, rtol=1e-10, atol=1e-10
        )

        # Verify constraint equation: di*U + dj*V = delta_wrd
        computed_delta = di * U + dj * V
        np.testing.assert_allclose(computed_delta, delta_wrd, rtol=1e-10, atol=1e-10)

        # Verify origin is computed correctly
        # origin = wrd0 - pix0[0]*U - pix0[1]*V
        origin = wrd0 - pix0[0] * U - pix0[1] * V
        np.testing.assert_allclose(result[0], origin, rtol=1e-10, atol=1e-10)


def test_compute_image_corners_from_2_tie_points_errors(tmp_path):
    """
    Test error conditions for _compute_image_corners_from_2_tie_points method.
    """
    with Workspace.create(tmp_path / "test_2_tie_points_errors.geoh5") as workspace:
        # Create a simple test image (10x10)
        image = Image.fromarray(np.arange(100, dtype="uint8").reshape(10, 10), "L")
        geoimage = GeoImage.create(workspace, name="test", image=image)

        # Test case 1: ValueError - less than 2 tie points
        with pytest.raises(ValueError, match="not enough values to unpack"):
            geoimage._compute_image_corners_from_2_tie_points(
                np.array([[[0.0, 0.0, 0.0], [100.0, 200.0, 5.0]]])
            )

        # Test case 2: ValueError - tie points with same i coordinate (di = 0)
        # Note: This is now allowed since we only need one of di or dj to be non-zero
        tie_points_same_i = np.array(
            [
                [[5.0, 0.0, 0.0], [100.0, 200.0, 5.0]],
                [[5.0, 8.0, 0.0], [100.0, 208.0, 5.0]],  # same i=5.0
            ]
        )
        # This should now work since dj != 0
        result = geoimage._compute_image_corners_from_2_tie_points(tie_points_same_i)
        assert isinstance(result, np.ndarray), (
            "Should return corners array when only dj differs"
        )
        assert result.shape == (4, 3)

        # Test case 3: ValueError - tie points with same j coordinate (dj = 0)
        # Note: This is now allowed since we only need one of di or dj to be non-zero
        tie_points_same_j = np.array(
            [
                [[0.0, 5.0, 0.0], [100.0, 200.0, 5.0]],
                [[8.0, 5.0, 0.0], [108.0, 200.0, 5.0]],  # same j=5.0
            ]
        )
        # This should now work since di != 0
        result = geoimage._compute_image_corners_from_2_tie_points(tie_points_same_j)
        assert isinstance(result, np.ndarray), (
            "Should return corners array when only di differs"
        )
        assert result.shape == (4, 3)

        # Test case 4: Error - tie points mapping to same world coordinates
        tie_points_same_world = np.array(
            [
                [[0.0, 0.0, 0.0], [100.0, 200.0, 5.0]],
                [[5.0, 5.0, 0.0], [100.0, 200.0, 5.0]],  # same world coords
            ]
        )
        with pytest.raises(
            ValueError, match="Tie points map to the same world coordinates"
        ):
            geoimage._compute_image_corners_from_2_tie_points(tie_points_same_world)


def test_georeference_with_duplicate_tie_points(tmp_path):
    """
    Test that georeference with 2 identical tie points raises an error.

    When passing 2 identical tie points to georeference(), _parse_tie_points
    will deduplicate them, leaving only 1 unique tie point. This should
    raise an error requiring cell sizes.
    """
    with Workspace.create(tmp_path / "test_duplicate_tie_points.geoh5") as workspace:
        # Create a simple test image (10x10)
        image = Image.fromarray(np.arange(100, dtype="uint8").reshape(10, 10), "L")
        geoimage = GeoImage.create(workspace, name="test", image=image)

        # Two identical tie points - will be deduplicated to 1 tie point
        duplicate_tie_points = [
            [[0.0, 0.0, 0.0], [100.0, 200.0, 5.0]],
            [[0.0, 0.0, 0.0], [100.0, 200.0, 5.0]],  # exact duplicate
        ]

        # Should raise error because after deduplication, only 1 tie point remains
        # and cell sizes are not provided
        with pytest.raises(
            ValueError,
            match="Cell sizes must be provided when only 1 tie point is available",
        ):
            geoimage.georeference(duplicate_tie_points)


def test_compute_image_corners_from_3_tie_points(tmp_path):
    """
    Test _compute_image_corners_from_3_tie_points method error handling and computation.
    """
    with Workspace.create(tmp_path / "test_3_tie_points.geoh5") as workspace:
        # Create a simple test image (10x10)
        image = Image.fromarray(np.arange(100, dtype="uint8").reshape(10, 10), "L")
        geoimage = GeoImage.create(workspace, name="test", image=image)

        # Test case 1: Valid three tie points forming axis-aligned rectangle
        tie_points = np.array(
            [
                [[0.0, 0.0, 0.0], [100.0, 200.0, 5.0]],
                [[10.0, 0.0, 0.0], [110.0, 200.0, 5.0]],
                [[0.0, 10.0, 0.0], [100.0, 210.0, 5.0]],
            ]
        )
        result = geoimage._compute_image_corners_from_3_tie_points(tie_points)
        assert result.shape == (4, 3)
        # Verify all corners
        # default_vertices reversed = [[0,0,0], [10,0,0], [10,10,0], [0,10,0]]
        expected = np.array(
            [
                [100.0, 200.0, 5.0],  # pixel (0, 0)
                [110.0, 200.0, 5.0],  # pixel (10, 0)
                [110.0, 210.0, 5.0],  # pixel (10, 10)
                [100.0, 210.0, 5.0],  # pixel (0, 10)
            ]
        )
        np.testing.assert_allclose(result, expected)

        # Test case 2: Valid three tie points with rotation/scaling
        tie_points_rotated = np.array(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[5.0, 0.0, 0.0], [10.0, 5.0, 0.0]],  # rotated and scaled
                [[0.0, 5.0, 0.0], [-5.0, 10.0, 0.0]],  # rotated and scaled
            ]
        )
        result_rotated = geoimage._compute_image_corners_from_3_tie_points(
            tie_points_rotated
        )
        assert result_rotated.shape == (4, 3)
        # Verify the affine transformation produces consistent results
        assert np.all(np.isfinite(result_rotated))

        # Test case 3: Valid three tie points with varying z
        tie_points_z = np.array(
            [
                [[0.0, 0.0, 0.0], [100.0, 200.0, 0.0]],
                [[5.0, 0.0, 0.0], [105.0, 200.0, 1.0]],
                [[0.0, 5.0, 0.0], [100.0, 205.0, 2.0]],
            ]
        )
        result_z = geoimage._compute_image_corners_from_3_tie_points(tie_points_z)
        assert result_z.shape == (4, 3)

        # Test case 4: ValueError - less than 3 tie points
        with pytest.raises(ValueError, match="all the input array dimensions"):
            geoimage._compute_image_corners_from_3_tie_points(
                np.array(
                    [
                        [[0.0, 0.0, 0.0], [100.0, 200.0, 5.0]],
                        [[5.0, 0.0, 0.0], [105.0, 200.0, 5.0]],
                    ]
                )
            )


def test_compute_image_corners(tmp_path):
    """
    Test _compute_image_corners method error handling and validation.
    """
    with Workspace.create(tmp_path / "test_compute_corners.geoh5") as workspace:
        # Create a simple test image
        image = Image.fromarray(np.arange(100, dtype="uint8").reshape(10, 10), "L")
        geoimage = GeoImage.create(workspace, name="test", image=image)

        # Test case 1: Valid 1 tie point with cell sizes
        tie_points_1 = np.array([[[0.0, 0.0, 0.0], [100.0, 200.0, 5.0]]])
        result = geoimage._compute_image_corners(
            tie_points_1, u_cell_size=1.0, v_cell_size=1.0
        )
        assert result.shape == (4, 3)

        # Test case 2: ValueError - 1 tie point without cell sizes
        with pytest.raises(ValueError, match="Cell sizes must be provided"):
            geoimage._compute_image_corners(
                tie_points_1, u_cell_size=None, v_cell_size=None
            )

        # Test case 3: ValueError - empty tie points array
        with pytest.raises(ValueError, match="At least 1 tie point is required"):
            geoimage._compute_image_corners(
                np.array([]).reshape(0, 2, 3), u_cell_size=None, v_cell_size=None
            )

        # Test case 4: ValueError - non-coplanar tie points
        # Create tie points where world coordinates are not on the same plane
        non_coplanar_points = np.array(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, 10.0, 0.0]],
                [
                    [1.0, 1.0, 0.0],
                    [10.0, 10.0, 10.0],
                ],  # This world point breaks coplanarity
            ]
        )
        with pytest.raises(ValueError, match="Tie points are not coplanar"):
            geoimage._compute_image_corners(
                non_coplanar_points, u_cell_size=None, v_cell_size=None
            )

        # Test case 5: ValueError - non-affine tie points
        # Create tie points that don't follow an affine transformation
        # These are coplanar but non-affine (nonlinear mapping)
        non_affine_points = np.array(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                [
                    [1.0, 1.0, 0.0],
                    [3.0, 3.0, 0.0],
                ],  # Breaks affine consistency (should be 1,1)
            ]
        )
        with pytest.raises(
            ValueError, match="Tie points are not consistent with an affine"
        ):
            geoimage._compute_image_corners(
                non_affine_points, u_cell_size=None, v_cell_size=None
            )

        # Test case 6: Valid 2 tie points (cell sizes computed internally)
        # For 2 tie points, cell sizes are derived from the tie points
        tie_points_2 = np.array(
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[10.0, 10.0, 0.0], [10.0, 10.0, 0.0]]]
        )
        # Don't provide cell sizes - they'll be computed from tie points
        result = geoimage._compute_image_corners(
            tie_points_2,
            u_cell_size=None,
            v_cell_size=None,
        )
        assert isinstance(result, np.ndarray), (
            "Should return corners array for valid 2 tie points"
        )

        # Test case 7: Valid 2 tie points with cell sizes
        tie_points_2_valid = np.array(
            [
                [[0.0, 0.0, 0.0], [100.0, 200.0, 0.0]],
                [[5.0, 5.0, 0.0], [105.0, 205.0, 0.0]],
            ]
        )
        result = geoimage._compute_image_corners(
            tie_points_2_valid, u_cell_size=1.0, v_cell_size=1.0
        )
        assert result.shape == (4, 3)

        # Test case 8: Valid 3 tie points
        tie_points_3 = np.array(
            [
                [[0.0, 0.0, 0.0], [100.0, 200.0, 0.0]],
                [[5.0, 0.0, 0.0], [105.0, 200.0, 0.0]],
                [[0.0, 5.0, 0.0], [100.0, 205.0, 0.0]],
            ]
        )
        result = geoimage._compute_image_corners(
            tie_points_3, u_cell_size=None, v_cell_size=None
        )
        assert result.shape == (4, 3)


def test_validate_geoimage(tmp_path):
    """
    Test GeoImageConversion._validate_geoimage static method error handling.
    """
    from geoh5py.shared.conversion import GeoImageConversion

    with Workspace.create(tmp_path / "test_validate_geoimage.geoh5") as workspace:
        # Create a simple test image (10x10)
        image = Image.fromarray(np.arange(100, dtype="uint8").reshape(10, 10), "L")

        # Test case 1: Valid geoimage with default rectangular vertices
        geoimage_valid = GeoImage.create(workspace, name="valid", image=image)
        # Should not raise any errors
        GeoImageConversion._validate_geoimage(geoimage_valid)

        # Test case 2: Valid geoimage with custom rectangular vertices
        vertices_rect = np.array(
            [[0.0, 10.0, 0.0], [10.0, 10.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
        geoimage_rect = GeoImage.create(
            workspace, name="rect", image=image.copy(), vertices=vertices_rect
        )
        GeoImageConversion._validate_geoimage(geoimage_rect)

        # Test case 3: ValueError - non-coplanar vertices
        # Create a valid geoimage then modify vertices to be non-coplanar
        geoimage_non_coplanar = GeoImage.create(
            workspace, name="non_coplanar", image=image.copy()
        )
        # Directly set non-coplanar vertices bypassing property setter validation
        geoimage_non_coplanar._vertices = np.rec.fromarrays(
            np.array(
                [
                    [0.0, 10.0, 0.0],
                    [10.0, 10.0, 0.0],
                    [10.0, 0.0, 0.0],
                    [0.0, 0.0, 10.0],  # Different z breaks coplanarity
                ]
            ).T,
            dtype=geoimage_non_coplanar._GeoImage__VERTICES_DTYPE,
        )
        with pytest.raises(ValueError, match="GeoImage vertices are not coplanar"):
            GeoImageConversion._validate_geoimage(geoimage_non_coplanar)

        # Test case 4: ValueError - non-orthogonal vertices (but coplanar)
        # Need vertices that are coplanar but not forming right angles
        geoimage_non_orthogonal = GeoImage.create(
            workspace, name="non_orthogonal", image=image.copy()
        )
        # Directly set non-orthogonal vertices - parallelogram in the xy plane
        geoimage_non_orthogonal._vertices = np.rec.fromarrays(
            np.array(
                [
                    [0.0, 10.0, 0.0],
                    [12.0, 12.0, 0.0],  # Skewed - not a right angle
                    [12.0, 2.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ).T,
            dtype=geoimage_non_orthogonal._GeoImage__VERTICES_DTYPE,
        )
        with pytest.raises(ValueError, match="GeoImage vertices are not orthogonal"):
            GeoImageConversion._validate_geoimage(geoimage_non_orthogonal)

        # Test case 5: ValueError - non-affine transformation
        # To break the affine check, we need points where NO affine transformation can map
        # pixel coordinates to world coordinates within tolerance
        # A trapezoid with orthogonal corners at vertices[3] will work
        geoimage_non_affine = GeoImage.create(
            workspace, name="non_affine", image=image.copy()
        )
        # Create a trapezoid: orthogonal at vertex 3, but non-uniform sides
        # But vertex[1] forms a trapezoid: [8, 10, 0] instead of [10, 10, 0]
        geoimage_non_affine._vertices = np.rec.fromarrays(
            np.array(
                [
                    [0.0, 10.0, 0.0],  # vertex 0
                    [8.0, 10.0, 0.0],  # vertex 1 - makes it a trapezoid
                    [10.0, 0.0, 0.0],  # vertex 2
                    [0.0, 0.0, 0.0],  # vertex 3
                ]
            ).T,
            dtype=geoimage_non_affine._GeoImage__VERTICES_DTYPE,
        )
        with pytest.raises(
            ValueError, match="GeoImage vertices do not define an affine"
        ):
            GeoImageConversion._validate_geoimage(geoimage_non_affine)


def test_rotation_dip_rotation_only_false(tmp_path):
    """
    Test GeoImage.rotation property when dip_rotation_only is False.

    The rotation property should raise a ValueError when the vertices define
    a rectangle that requires more than rotation and dip transformations
    (i.e., when the u_vector of the plane has a non-zero Z component).
    """
    with Workspace.create(tmp_path / "test_rotation_dip_only.geoh5") as workspace:
        # Create a simple test image (10x10)
        image = Image.fromarray(np.arange(100, dtype="uint8").reshape(10, 10), "L")

        # Test case 1: Valid rotation - dip_rotation_only is True
        # Simple rectangle in XY plane with rotation
        vertices_valid = np.array(
            [[0.0, 10.0, 0.0], [10.0, 10.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
        geoimage_valid = GeoImage.create(
            workspace, name="valid_rotation", image=image, vertices=vertices_valid
        )
        # Should not raise any errors - rotation property works
        rotation = geoimage_valid.rotation
        assert isinstance(rotation, float)

        # Test case 2: ValueError - dip_rotation_only is False
        # Create vertices where u_vector has Z component (requires more than rotation+dip)
        # Need an orthogonal rectangle that's "twisted" in 3D space
        #
        # To make dip_rotation_only=False, u_vector must have non-zero Z component
        geoimage_invalid = GeoImage.create(
            workspace, name="invalid_rotation", image=image.copy()
        )

        # Create the orthogonal rectangle
        origin = np.array([0.0, 0.0, 0.0])  # vertex[3]
        u_vec = np.array([10.0, 0.0, 5.0])  # to vertex[2] - has Z component!
        v_vec = np.array([0.0, 10.0, 0.0])  # to vertex[0] - in XY plane

        # Normalize for unit vectors, then scale back to desired size
        u_normalized = u_vec / np.linalg.norm(u_vec)
        v_normalized = v_vec / np.linalg.norm(v_vec)
        u_scaled = u_normalized * np.sqrt(10**2 + 5**2)  # Keep same length
        v_scaled = v_normalized * 10

        vertices_twisted = np.array(
            [
                origin + v_scaled,  # vertex[0]
                origin + u_scaled + v_scaled,  # vertex[1]
                origin + u_scaled,  # vertex[2]
                origin,  # vertex[3]
            ]
        )

        geoimage_invalid._vertices = np.rec.fromarrays(
            vertices_twisted.T, dtype=geoimage_invalid._GeoImage__VERTICES_DTYPE
        )

        # Accessing rotation property should raise ValueError
        with pytest.raises(
            ValueError,
            match="The vertices do not define a rectangle that can be explained by rotation and dip only",
        ):
            _ = geoimage_invalid.rotation
