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

import numpy as np
import pytest

from geoh5py.objects import Grid2D
from geoh5py.workspace import Workspace


def test_attribute_setters():
    with Workspace() as workspace:
        grid = Grid2D.create(workspace)

        with pytest.raises(TypeError, match="Rotation angle must be a float"):
            grid.rotation = np.r_[0, 1]

        with pytest.raises(TypeError, match="Attribute 'origin' must be a list"):
            grid.origin = "abc"

        with pytest.raises(
            ValueError, match="Attribute 'origin' must be a list or array"
        ):
            grid.origin = np.r_[0, 1]

        with pytest.raises(ValueError, match="Array of 'origin' must be of dtype"):
            grid.origin = np.asarray(
                [0, 0, 0], dtype=np.dtype([("x", int), ("y", int), ("z", int)])
            )
