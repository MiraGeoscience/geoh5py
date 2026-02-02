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

from geoh5py.objects.maxwell_plate import MaxwellPlate, PlateGeometry, PlatePosition
from geoh5py.workspace import Workspace


# import pytest
TEST_PARAMETERS = {
    "position": {
        "x": 100.0,
        "y": 200.0,
        "z": -50.0,
    },
    "width": 300.0,
    "thickness": 20.0,
    "length": 45.0,
    "dip": 30.0,
}


def test_plate_position_validation():
    input_str = "100.0;{200.00,300.00,-400.00}"
    # Test valid parameters
    position = PlatePosition.model_validate(input_str)
    assert isinstance(position, PlatePosition)

    assert position.x == 200.0
    assert not position._initialized

    # Test invalid parameters
    invalid_params = TEST_PARAMETERS["position"].copy()
    invalid_params["x"] = "invalid"  # Invalid x position
    with pytest.raises(ValidationError):
        PlatePosition.model_validate(invalid_params)

    str_rep = position.model_dump()
    assert isinstance(str_rep, str)

    assert str_rep == input_str


def test_plate_geometry_validation():
    # Test valid parameters
    geometry = PlateGeometry.model_validate(TEST_PARAMETERS)
    assert isinstance(geometry, PlateGeometry)
    assert not geometry._initialized

    # Test invalid parameters
    invalid_params = TEST_PARAMETERS.copy()
    invalid_params["width"] = -100.0  # Invalid width
    with pytest.raises(ValueError):
        PlateGeometry.model_validate(invalid_params)

    invalid_params = TEST_PARAMETERS.copy()
    invalid_params["position"]["z"] = "invalid"  # Invalid z position
    with pytest.raises(ValidationError):
        PlateGeometry.model_validate(invalid_params)


def test_create_plate(tmp_path):
    filepath = tmp_path / f"{__name__}_test.geoh5"
    with Workspace.create(filepath) as ws:
        plate = MaxwellPlate.create(
            ws, geometry=PlateGeometry.model_validate(TEST_PARAMETERS)
        )
        assert isinstance(plate.geometry, PlateGeometry)
        assert isinstance(plate, MaxwellPlate)

    # Reopen and check values
    with Workspace(filepath) as ws:
        plate = ws.objects[0]
        for key, val in TEST_PARAMETERS.items():
            plate_val = getattr(plate.geometry, key)
            if isinstance(plate_val, PlatePosition):
                for coord in ["x", "y", "z"]:
                    assert np.isclose(getattr(plate_val, coord), val.get(coord))
            else:
                assert np.isclose(plate_val, val)

        # Change another value and save
        plate.geometry.dip_direction = 90
        plate.geometry.position.increment = 20

    # Reopen and check updated value
    with Workspace(filepath) as ws:
        plate = ws.objects[0]
        assert np.isclose(plate.geometry.dip_direction, 90)
        assert np.isclose(plate.geometry.position.increment, 20)
