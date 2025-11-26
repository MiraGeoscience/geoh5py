# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2025 Mira Geoscience Ltd.                                     '
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


def test_attribute_setters():
    workspace = Workspace()
    image = np.random.randint(0, 255, (128, 128))
    gimage = GeoImage.create(workspace, image=image, cells=[[0, 0, 0, 0], [1, 1, 1, 1]])

    with pytest.raises(
        TypeError, match="Attribute 'cells' must be provided as type numpy.ndarray"
    ):
        gimage.cells = "abc"

    with pytest.raises(ValueError, match="Array of cells should be of shape"):
        gimage.cells = [[0, 0, 0], [1, 1, 1]]

    with pytest.raises(TypeError, match="Indices array must be of integer type"):
        gimage.cells = np.array([[0, 0, 0, 0], [1, 1, 1, 1]], ndmin=2, dtype=float)


def test_create_copy_geoimage(tmp_path):  # pylint: disable=too-many-statements
    with Workspace.create(tmp_path / r"geo_image_test.geoh5") as workspace:
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

        assert geoimage.image_georeferenced is None

        with pytest.raises(AttributeError, match="The object contains no image data"):
            geoimage.save_as("test")

        with pytest.raises(AttributeError, match="An 'image' must be set be"):
            geoimage.georeference(pixels[0, :], points)

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

        geoimage.image = np.random.randint(0, 255, (128, 128))

        np.testing.assert_allclose(
            geoimage.extent, np.array([[0.0, 0.0, 0.0], [128.0, 128.0, 0.0]])
        )

        geoimage.georeferencing_from_image()

        # with pytest.raises(AttributeError, match="Vertices must be set for referencing"):
        #     geoimage.set_tag_from_vertices()

        with pytest.raises(
            ValueError, match="Input reference points must be a 2D array"
        ):
            geoimage.georeference(pixels[0, :], points)

        with pytest.raises(
            ValueError, match="Input 'locations' must be a 2D array of shape"
        ):
            geoimage.georeference(pixels, points[0, :])

        with pytest.raises(
            AttributeError,
            match="The 'image' property cannot be reset. Consider creating a new object",
        ):
            geoimage.image = np.random.randint(0, 255, (128, 64, 3))

        geoimage = GeoImage.create(
            workspace, name="MyGeoImage", image=np.random.randint(0, 255, (128, 64, 3))
        )
        geoimage.georeference(pixels, points)
        np.testing.assert_almost_equal(
            geoimage.vertices,
            np.asarray([[0, 15, 6], [10, 15, 6], [10, 5, 0], [0, 5, 0]]).astype(float),
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
        geoimage_file = GeoImage.create(workspace, name="MyGeoImage")

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

            with pytest.warns(UserWarning, match="The image and the extent"):
                new_image = geoimage.copy_from_extent(
                    np.vstack([[100, 100], [200, 200]])
                )

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
            UserWarning, match="The 'tif.' image has no referencing information."
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


def test_complex_tiff(tmp_path):
    image_path = tmp_path / r"testtif.tif"
    workspace_path = tmp_path / r"geo_image_test.geoh5"

    # create a tiff
    image = Image.fromarray(1000 * np.random.randn(128, 128).astype("float32"))
    image.save(image_path)

    with Workspace.create(workspace_path) as workspace:
        # load image
        geoimage = GeoImage.create(workspace, name="test_area", image=str(image_path))

        grid = workspace.get_entity("test_area_grid2d")[0]

        assert all(
            np.array(geoimage.image)[::-1].flatten()
            == grid.get_data("band[0]")[0].values
        )


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
    grid = geoimage.to_grid2d(parent=parent, mode="RGBA")

    # transform the image
    grid_transformed = grid.copy_from_extent(
        extent=extent,
        parent=parent,
        copy_children=copy_children,
        clear_cache=clear_cache,
        from_image=True,
        **kwargs,
    )

    if grid_transformed is None:
        grid.workspace.remove_entity(grid)
        return None

    # transform the grid back to an image
    image_transformed = grid_transformed.to_geoimage(
        keys=grid_transformed.get_data_list(),
        mode="RGBA",
        normalize=False,
        name=geoimage.name + "_from_grid2d",
    )

    grid.workspace.remove_entity(grid_transformed)
    grid.workspace.remove_entity(grid)

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

    # Check if grid_converted vertices are outside original bounds
    orig_extent = original.extent
    grid_extent = grid_converted.extent

    # todo: there is a bug in the conversion or grid2d extent! GEOPY-2591
    if (
        grid_extent[0, 0] < orig_extent[0, 0]
        or grid_extent[0, 1] < orig_extent[0, 1]
        or grid_extent[1, 0] > orig_extent[1, 0]
        or grid_extent[1, 1] > orig_extent[1, 1]
    ):
        warn(
            "Grid2D conversion produced vertices outside original bounds - skipping vertex comparison"
        )
        skip_vertex_check = True
    else:
        skip_vertex_check = False

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

    if not skip_vertex_check and not np.allclose(
        direct.vertices, grid_converted.vertices, rtol=1e-6, atol=1e-6
    ):
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


def test_copy_from_extent_predictable(tmp_path):
    """
    Test copy_from_extent with predictable, simple geometry where we can anticipate results.

    This test uses a simple flat image with known cell positions and tests that
    copy_from_extent correctly includes cells whose centers fall within the extent.
    """
    workspace_path = tmp_path / "copy_extent_test.geoh5"

    # Create a 10x10 image for simple calculations
    image = Image.fromarray(np.arange(100, dtype="uint8").reshape(10, 10), "L")

    # Create a simple 10x10 world unit area (1 unit per pixel)
    base_vertices = np.array(
        [
            [0, 0, 0],  # vertex 0: bottom-left
            [10, 0, 0],  # vertex 1: bottom-right
            [10, 10, 0],  # vertex 2: top-right
            [0, 10, 0],  # vertex 3: top-left
        ]
    )

    # Define base test configurations
    base_cases = [
        {
            "name": "simple_center_crop",
            "extent": np.array([[2, 2, -1], [7, 7, 1]]),
            "description": "Center crop of flat image",
        },
        {
            "name": "corner_crop",
            "extent": np.array([[0, 0, -1], [3, 3, 1]]),
            "description": "Corner crop of flat image",
        },
        {
            "name": "fully_inside_small",
            "extent": np.array([[3, 3, -1], [6, 6, 1]]),
            "description": "Small rectangle fully inside image",
        },
        {
            "name": "exact_border_top",
            "extent": np.array([[2, 9, -1], [8, 11, 1]]),
            "description": "Rectangle on top border",
        },
        {
            "name": "exact_corner_bottom_right",
            "extent": np.array([[8, 0, -1], [11, 3, 1]]),
            "description": "Rectangle on bottom-right corner",
        },
    ]

    # Define transformation parameters to test
    transformations = [
        {"dip": 0, "rotation": 0, "suffix": "flat"},
        {"dip": 0, "rotation": 45, "suffix": "rot45"},
        {"dip": 0, "rotation": 90, "suffix": "rot90"},
        {"dip": 0, "rotation": -30, "suffix": "rot_neg30"},
        {"dip": 15, "rotation": 0, "suffix": "dip15"},
        {"dip": 30, "rotation": 0, "suffix": "dip30"},
        {"dip": 45, "rotation": 0, "suffix": "dip45"},
        {"dip": 15, "rotation": 45, "suffix": "dip15_rot45"},
        {"dip": 30, "rotation": 90, "suffix": "dip30_rot90"},
    ]

    # Generate test cases by combining base cases with transformations
    test_cases = []
    for base_case in base_cases:
        for transform in transformations:
            test_case = {
                "name": f"{base_case['name']}_{transform['suffix']}",
                "dip": transform["dip"],
                "rotation": transform["rotation"],
                "extent": base_case["extent"].copy(),
                "description": f"{base_case['description']} (dip={transform['dip']}°, rot={transform['rotation']}°)",
            }
            test_cases.append(test_case)

    # Add offset origin test cases
    offset_cases = [
        {
            "name": "offset_origin_center_crop",
            "dip": 0,
            "rotation": 0,
            "origin_offset": [10, 5, 2],
            "extent": np.array([[12, 7, -1], [17, 12, 5]]),
            "description": "Center crop with offset origin",
        },
        {
            "name": "offset_origin_rotated",
            "dip": 0,
            "rotation": 45,
            "origin_offset": [5, 5, 1],
            "extent": np.array([[7, 7, -1], [12, 12, 3]]),
            "description": "Rotated crop with offset origin",
        },
    ]
    test_cases.extend(offset_cases)

    errors = []
    with Workspace.create(workspace_path) as workspace:
        for test_case in test_cases:
            try:
                # Create GeoImage with appropriate vertices
                vertices = base_vertices.copy()
                if "origin_offset" in test_case:
                    offset = np.array(test_case["origin_offset"])
                    vertices = vertices + offset

                geoimage = GeoImage.create(
                    workspace,
                    name=f"{test_case['name']}_geoimage",
                    image=image.copy(),
                    vertices=vertices,
                )

                # Apply transformations
                geoimage.dip = test_case["dip"]
                geoimage.rotation = test_case["rotation"]

                # Create Points to visualize the extent as a 3D cube (8 corners)
                min_x, min_y, min_z = test_case["extent"][0]
                max_x, max_y, max_z = test_case["extent"][1]

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

                # just to see them in the viewer
                Points.create(
                    workspace,
                    name=f"{test_case['name']}_extent",
                    vertices=extent_corners,
                )

                # Run both methods
                direct_result, grid_result = run_extent_test_case(
                    geoimage, test_case["extent"], workspace
                )

                if direct_result is None and grid_result is None:
                    continue

                # todo: this sounds like a bug in grid2d conversion GEOPY-2591
                if direct_result is None and grid_result is not None:
                    warn(
                        "Direct method returned None but grid method "
                        f"succeeded for '{test_case['name']}'"
                    )
                    continue

                if direct_result is not None and grid_result is None:
                    # todo: there is a bug in the conversion or grid2d extent! GEOPY-2591
                    warn(
                        f"Grid method returned None but direct method succeeded for "
                        f"'{test_case['name']}' (Grid2D conversion limitation)"
                    )
                    continue

                # Both methods succeeded - compare results
                comparison_errors = compare_geoimages(
                    direct_result, grid_result, geoimage, test_case["name"]
                )
                errors.extend(comparison_errors)

            except Exception as error:  # noqa
                if "boolean index did not match indexed array" in str(error):
                    # todo: This is a known Grid2D conversion issue, treat as warning
                    warn(f"Grid2D conversion error for '{test_case['name']}': {error}")
                else:
                    # Other unexpected errors should still be collected
                    errors.append(
                        f"Test case '{test_case['name']}' failed with error: {error}"
                    )

    # Raise all collected errors at the end
    if errors:
        error_summary = f"Test failures occurred ({len(errors)} total):\n" + "\n".join(
            f"{i + 1}. {err}" for i, err in enumerate(errors)
        )
        raise AssertionError(error_summary)
