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

from geoh5py.objects.maxwell_plate import MaxwellPlate, PlateGeometry
from geoh5py.workspace import Workspace


# import pytest


def test_read_file(tmp_path):
    h5file_path = r"C:\Users\dominiquef\Desktop\maxwell_plate.geoh5"

    with Workspace(h5file_path) as ws:
        viz = ws.objects[0].visual_parameters
        xml = viz.xml

        options = {child.tag: child.text for child in xml}
        geometry = PlateGeometry(parent=viz, **options)
        print(geometry.position.x)
        geometry.width = 123.0
        geometry.position.x = geometry.position.x + 10.0


def test_create_plate(tmp_path):
    filepath = tmp_path / "maxwell_plate_test.geoh5"
    with Workspace.create(filepath) as ws:
        plate = MaxwellPlate.create(
            ws,
            name="Test Plate",
            position={
                "x": 100.0,
                "y": 200.0,
                "z": -50.0,
            },
            width=300.0,
            height=150.0,
            thickness=20.0,
            length=45.0,
            dip=30.0,
        )
        assert isinstance(plate, MaxwellPlate)
        assert plate.visual_parameters.get_child("position_x").value == 100.0
        assert plate.visual_parameters.get_child("width").value == 300.0
