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

import random
import string
from pathlib import Path

import numpy as np

from geoh5py.objects import TextObject
from geoh5py.objects.text import TextData
from geoh5py.workspace import Workspace


def test_create_text_object(tmp_path: Path):
    with Workspace.create(tmp_path / f"{__name__}.geoh5") as workspace:
        # Create a text object
        # ws.objects[0].copy(parent=workspace)
        xyz = np.random.rand(6, 3) * 10
        random_labels = [
            "".join(
                random.choice(string.ascii_lowercase)
                for _ in range(np.random.randint(1, 10))
            )
            for _ in range(xyz.shape[0])
        ]
        color = "#ff00f1ff"

        text = TextObject.create(
            workspace, vertices=xyz, text=random_labels, name="Text Object", color=color
        )

        assert isinstance(text.text_mesh_data, TextData)
        for entry, label in zip(
            text.text_mesh_data.text_data, random_labels, strict=False
        ):
            assert entry.text == label
            assert entry.color == color
