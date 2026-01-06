# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2025-2026 Mira Geoscience Ltd.                                '
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

import logging
from pathlib import Path

import numpy as np

from geoh5py.objects import Label
from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace


def test_create_label(tmp_path: Path, caplog):
    h5file_path = tmp_path / r"testGroup.geoh5"

    # Create a workspace
    with Workspace.create(h5file_path) as workspace:
        label = Label.create(workspace, name="MyTestLabel")

        assert label.copy_from_extent(np.vstack([[0, 0], [1, 1]])) is None

        with caplog.at_level(logging.WARNING):
            copy_label = label.copy(mask=[[0, 0], [1, 1]])

        assert "Masking is not supported" in caplog.text

        compare_entities(
            label, copy_label, ignore=["target_position", "label_position", "_uid"]
        )
