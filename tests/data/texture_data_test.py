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

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.spatial import Delaunay

from geoh5py.data.texture_data import CompressedTextures
from geoh5py.objects import Grid2D, Surface
from geoh5py.workspace import Workspace


def create_texture(workspace, image_size=(8, 16)):
    u_pixel, v_pixel = np.meshgrid(
        np.arange(image_size[1], dtype=float), np.arange(image_size[0], dtype=float)
    )
    image = u_pixel + v_pixel * image_size[0]
    image = np.dstack([image, image, image])
    u_pixel = u_pixel.flatten()
    u_pixel /= image_size[1]
    u_pixel += 1 / image_size[1] / 2
    v_pixel = v_pixel.flatten()
    v_pixel /= image_size[0]
    v_pixel += 1 / image_size[0] / 2
    pixels = np.c_[u_pixel, v_pixel]
    x_locs, y_locs = np.meshgrid(np.arange(image_size[1]), np.arange(image_size[0]))
    vertices = np.c_[
        x_locs.flatten(),
        y_locs.flatten(),
        (
            np.sin(y_locs / y_locs.max() * np.pi)
            * np.sin(x_locs / x_locs.max() * np.pi)
        ).flatten(),
    ]
    surf = Delaunay(vertices[:, :2])
    obj = Surface.create(
        workspace,
        vertices=vertices,
        cells=surf.simplices,
    )

    texture = obj.add_data(
        {
            "test_texture": {
                "primitive_type": "TEXTURE",
                "association": "VERTEX",
            },
        }
    )
    return texture, image, pixels


def test_create_texture(tmp_path):
    with Workspace.create(tmp_path / f"{__name__}.geoh5") as workspace:
        texture, image, pixels = create_texture(workspace)

        with pytest.raises(
            ValueError, match="Shape of the 'texture_image' must be a 2D"
        ):
            texture.texture_image = image.flatten()

        with pytest.raises(TypeError, match="Attribute 'values' must be a list"):
            texture.values = "abc"

        with pytest.raises(ValueError, match="'values' requires an ndarray of shape"):
            texture.values = np.array([1, 2, 3])

        with pytest.raises(TypeError, match="Array of 'values' must be of dtype"):
            texture.values = np.asarray(
                np.rec.fromarrays(((1, 2), (3, 4)), dtype=[("a", int), ("b", int)])
            )

        with pytest.raises(ValueError, match="The length of 'values'"):
            texture.values = np.asarray(
                np.rec.fromarrays(
                    ((1, 2), (3, 4)), dtype=np.dtype([("v[0]", "<f4"), ("v[1]", "<f4")])
                )
            )

        texture.values = pixels
        texture.texture_image = image

        grid = Grid2D.create(workspace)

        with pytest.raises(
            TypeError, match="The parent of `texture_data` must have vertices"
        ):
            texture.copy(parent=grid)

    # Re-open and check the texture
    with Workspace(tmp_path / f"{__name__}.geoh5") as workspace:
        texture = workspace.get_entity("test_texture")[0]
        np.testing.assert_almost_equal(
            np.asarray(texture.image), (image / image.max() * 255).astype(int)
        )


def test_compressed_textures(tmp_path):
    file = tmp_path / f"{__name__}.geoh5"

    with Workspace.create(file) as workspace:
        texture, image, pixels = create_texture(workspace, image_size=(7, 15))

        with pytest.raises(TypeError, match="must be a numpy array or PIL"):
            CompressedTextures.get_padded_image([1, 2])

        with pytest.raises(ValueError, match="must be a 2D or a 3D array"):
            CompressedTextures.get_padded_image(image.flatten())

        padded = CompressedTextures.get_padded_image(image)
        valid_widths = image.shape[1]
        valid_heights = image.shape[0]

        width, height = padded.shape[1], padded.shape[0]
        CompressedTextures.get_padded_image(image)
        texture_kwargs = {
            "valid_widths": np.r_[valid_widths],
            "valid_heights": np.r_[valid_heights],
            "widths": np.r_[width],
            "heights": np.r_[height],
            "textures": {"Blocks_0": padded},
        }

        with pytest.raises(ValidationError, match="Formats must be one"):
            texture_kwargs["formats"] = np.r_[123]
            CompressedTextures(**texture_kwargs)

        with pytest.raises(
            ValidationError, match="All arrays must have the same length"
        ):
            texture_kwargs["formats"] = np.r_[32849, 32849]
            CompressedTextures(**texture_kwargs)

        with pytest.raises(
            TypeError, match="must be a dict, CompressedTextures or None"
        ):
            texture.compressed_textures = "abc"

        texture_kwargs["formats"] = np.r_[32849]
        compressed_texture = CompressedTextures(**texture_kwargs)

        pixels = np.c_[pixels, np.zeros((pixels.shape[0], 1))]
        texture.values = pixels
        texture.compressed_textures = compressed_texture

    with Workspace(file) as workspace:
        texture = workspace.get_entity("test_texture")[0]
        assert texture.compressed_textures is not None
        assert compressed_texture.valid_widths[0] == valid_widths
        assert compressed_texture.valid_heights[0] == valid_heights
        assert compressed_texture.widths[0] == width
        assert compressed_texture.heights[0] == height
        assert compressed_texture.formats[0] == 32849
