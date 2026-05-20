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
from PIL import Image
from PIL.TiffImagePlugin import TiffImageFile
from scipy.spatial import Delaunay

from geoh5py.objects import Surface
from geoh5py.workspace import Workspace


def test_create_texture(tmp_path):
    workspace = Workspace.create(tmp_path / f"{__name__}.geoh5")
    image = np.random.randint(0, 255, (128, 128))

    u_pixel, v_pixel = np.meshgrid(
        np.arange(image.shape[1], dtype=float), np.arange(image.shape[0], dtype=float)
    )
    u_pixel = u_pixel.flatten()
    u_pixel /= image.shape[0]
    u_pixel += 1 / image.shape[0] / 2
    v_pixel = v_pixel.flatten()
    v_pixel /= image.shape[1]
    v_pixel += 1 / image.shape[1] / 2

    x_locs, y_locs = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
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
                "values": np.c_[u_pixel, v_pixel],
                "primitive_type": "TEXTURE",
                "association": "VERTEX",
                "texture_image": image,
            },
        }
    )
    assert texture.image


def test_load_file():
    file = r"C:\Users\dominiquef\Downloads\texture_2d_data_obj_file.geoh5"
    workspace = Workspace(file)
    texture = workspace.get_entity("Tile_6_0")[0]
    assert texture.image
    assert texture.values is not None
