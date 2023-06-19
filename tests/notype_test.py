#  Copyright (c) 2023 Mira Geoscience Ltd.
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

from geoh5py.objects import NoTypeObject
from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace


def test_create_notype(tmp_path):
    h5file_path = tmp_path / r"testNoType.geoh5"

    # Create a workspace
    with Workspace(h5file_path) as workspace:
        label = NoTypeObject.create(workspace, name="MyTestLabel")

        assert label.copy_from_extent(np.vstack([[0, 0], [1, 1]])) is None

        with pytest.warns(UserWarning):
            copy_label = label.copy(mask=[[0, 0], [1, 1]])
        compare_entities(label, copy_label, ignore=["_uid"])
