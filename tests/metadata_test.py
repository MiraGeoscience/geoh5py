#  Copyright (c) 2024 Mira Geoscience Ltd.
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

from geoh5py.objects import Points
from geoh5py.workspace import Workspace


def test_metadata(tmp_path):
    h5file_path = tmp_path / r"testPoints.geoh5"
    workspace = Workspace.create(h5file_path)
    points = Points.create(workspace, vertices=np.random.randn(12, 3), allow_move=False)

    assert points.metadata is None

    points.metadata = {"test": "test"}

    assert points.metadata == {"test": "test"}

    points.metadata = {"test2": "test"}

    assert points.metadata == {"test2": "test"}

    points.update_metadata({"test3": "test3"})

    assert points.metadata == {"test2": "test", "test3": "test3"}

    points.metadata = None

    assert points.metadata is None

    with pytest.raises(TypeError, match="Input metadata must be of type"):
        points.metadata = "bidon"

    with pytest.raises(TypeError, match="Input metadata must be of type"):
        points.update_metadata("bidon")
