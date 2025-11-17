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
import re
import string
from pathlib import Path

import numpy as np
import pytest
from pydantic_core import ValidationError

from geoh5py.objects import TextObject
from geoh5py.objects.text import TextMesh
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

        assert isinstance(text.text_mesh_data, TextMesh)
        for entry, label in zip(text.text_mesh_data.data, random_labels, strict=False):
            assert entry.text == label
            assert entry.color == color


def test_text_data_length_mismatch(tmp_path):
    with Workspace.create(tmp_path / "test.geoh5") as workspace:
        # Create a TextObject with 5 vertices
        vertices = np.random.rand(5, 3)
        text_object = TextObject.create(workspace, vertices=vertices)

        with pytest.raises(ValidationError, match="Field required"):
            text_object.text_mesh_data = '{"abc":{"label": "a"}}'

        # Test setting text_mesh_data with bad type
        with pytest.raises(
            TypeError,
            match=re.escape("The 'Text Data' must be a dictionary or a JSON string."),
        ):
            text_object.text_mesh_data = 123

        # Test setting value to the text_mesh_data property through setattr
        with pytest.raises(
            ValueError,
            match="entries must contain a list of len",
        ):
            text_object.color = ["#ff00f1ff", "#00ff1fff", "#0000ffff"]

        text_object.direction = ([1.5, 0.5, 0.5],)

        assert text_object.text_mesh_data.data[0].direction == "{1.5,0.5,0.5}"

        # Create invalid text_mesh_data with mismatched length
        invalid_text_mesh_data = {
            "data": [
                {"text": "Label1", "starting_point": "{0,0,0}"},
                {"text": "Label2", "starting_point": "{1,1,1}"},
            ]
        }

        with pytest.raises(
            ValueError,
            match=re.escape(
                "The 'Text Data' dictionary must contain a list of len\\('n_vertices'\\)."
            ),
        ):
            text_object.text_mesh_data = invalid_text_mesh_data


def test_copy_extent(tmp_path):
    with Workspace.create(tmp_path / "test.geoh5") as workspace:
        # Create a TextObject with 5 vertices
        vertices = np.random.rand(5, 3)
        text_object = TextObject.create(workspace, vertices=vertices)

        # Create a copy of the TextObject
        copied_text_object = text_object.copy_from_extent(
            np.c_[[-np.inf, np.inf], [-np.inf, np.inf]]
        )

        # Check if the copied object has the same extent as the original
        assert np.array_equal(
            copied_text_object.extent, text_object.extent, equal_nan=True
        )

        # Test copy by mask
        mask = np.array([True, False, True, False, True])
        new_copy = text_object.copy(mask=mask)

        assert len(new_copy.text_mesh_data.data) == 3
