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

from geoh5py.objects import GeoImage
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

DEFAULT_CRS = {"Code": "Unknown", "Name": "Unknown"}


def test_coordinate_system(tmp_path):
    workspace = Workspace.create(tmp_path / r"geo_image_test.geoh5")

    # create and save a tiff
    image = Image.fromarray(np.random.randint(0, 255, (128, 128)), "RGB")

    for id_ in tag.items():
        image.getexif()[id_[0]] = id_[1]

    image.save(tmp_path / r"testtif.tif", exif=image.getexif())

    # load image
    geoimage = GeoImage.create(
        workspace, name="test_area_a", image=f"{tmp_path!s}/testtif.tif"
    )

    # create RGB grid2d
    grid2d_rgb = geoimage.to_grid2d(new_name="RGB")

    # verify the default coordinate system
    assert grid2d_rgb.coordinate_reference_system == DEFAULT_CRS

    # set reference coordinate system
    coordinate_system = {
        "Name": "WGS 84 / UTM zone 34N",
        "Code": "EPSG:32634",
    }

    grid2d_rgb.coordinate_reference_system = coordinate_system

    assert grid2d_rgb.coordinate_reference_system == coordinate_system

    grid2 = grid2d_rgb.copy()

    # set reference coordinate system
    coordinate_system = {
        "Code": "bidon",
        "Name": "mais vraiment bidon",
    }

    grid2.coordinate_reference_system = coordinate_system

    with pytest.raises(
        TypeError, match="Input coordinate reference system must be a dictionary"
    ):
        grid2.coordinate_reference_system = "bidon"

    with pytest.raises(
        KeyError,
        match="Input coordinate reference system must only contain a 'Code' and 'Name' keys",
    ):
        grid2.coordinate_reference_system = {"Bidon": "bidon"}
