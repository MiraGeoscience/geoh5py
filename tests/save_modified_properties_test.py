#  Copyright (c) 2021 Mira Geoscience Ltd.
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

import tempfile
from pathlib import Path
from unittest.mock import PropertyMock, patch

import numpy as np

from geoh5py.objects import Points
from geoh5py.workspace import Workspace


def test_save_modified_properties():
    n_data = 12
    xyz = np.random.randn(n_data, 3)

    with tempfile.TemporaryDirectory() as tempdir:
        h5file_path = Path(tempdir) / r"testPoints.geoh5"

        # Create a workspace
        workspace = Workspace(h5file_path)

        with patch(
            "geoh5py.shared.Entity.allow_move", new_callable=PropertyMock
        ) as allow_move:
            allow_move.return_value = True
            points = Points.create(workspace, vertices=xyz)
            points.vertices = xyz * 2

            workspace.finalize()

            assert (
                len(allow_move.mock_calls) == 1
            ), "Attributes re-written along with vertices."

            points.allow_delete = False

            workspace.finalize()

            assert (
                len(allow_move.mock_calls) == 2
            ), "Not all attributes were re-written."
