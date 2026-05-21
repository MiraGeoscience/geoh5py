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
from scipy.spatial import Delaunay

from geoh5py.objects import Surface
from geoh5py.workspace import Workspace


def test_create_texture(tmp_path):
    with Workspace.create(tmp_path / f"{__name__}.geoh5") as workspace:
        image_size = 8, 16

        u_pixel, v_pixel = np.meshgrid(
            np.arange(image_size[1], dtype=float), np.arange(image_size[0], dtype=float)
        )
        image = u_pixel + v_pixel * image_size[0]
        u_pixel = u_pixel.flatten()
        u_pixel /= image_size[1]
        u_pixel += 1 / image_size[1] / 2
        v_pixel = v_pixel.flatten()
        v_pixel /= image_size[0]
        v_pixel += 1 / image_size[0] / 2

        x_locs, y_locs = np.meshgrid(np.arange(image_size[1]), np.arange(image_size[0]))
        vertices = np.c_[
            x_locs.flatten(), y_locs.flatten(), np.zeros_like(y_locs).flatten()
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

        texture.values = np.c_[u_pixel, v_pixel]
        texture.texture_image = image

    # Re-open and check the texture
    with Workspace(tmp_path / f"{__name__}.geoh5") as workspace:
        texture = workspace.get_entity("test_texture")[0]
        np.testing.assert_almost_equal(
            np.asarray(texture.image), (image / image.max() * 255).astype(int)
        )
