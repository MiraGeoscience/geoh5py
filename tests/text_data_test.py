# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2020-2026 Mira Geoscience Ltd.                                     '
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

import h5py
import numpy as np
import pytest

from geoh5py.objects import Points
from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace


def test_create_text_data(tmp_path: Path):
    name = "MyTestPointset"

    # Generate a random cloud of points with reference values
    n_data = 12
    values = np.asarray(
        [
            "".join(random.choice(string.ascii_lowercase) for i in range(8))
            for jj in range(12)
        ]
    )

    h5file_path = tmp_path / r"testTextData.geoh5"

    with Workspace.create(h5file_path) as workspace:
        with Workspace.create(tmp_path / r"testTextData_copy.geoh5") as new_workspace:
            points = Points.create(
                workspace,
                vertices=np.random.randn(n_data, 3),
                name=name,
                allow_move=False,
            )

            data = points.add_data(
                {
                    "DataValues": {
                        "type": "text",
                        "values": values,
                    }
                }
            )

            word = points.add_data(
                {
                    "WordValue": {
                        "type": "text",
                        "values": values[0],
                    }
                }
            )

            points.copy(new_workspace)

            with pytest.raises(ValueError, match="Input 'values' for"):
                points.add_data(
                    {
                        "bidon": {
                            "type": "text",
                            "values": np.array([1236547])[0],
                        }
                    }
                )

    with workspace.open():
        with new_workspace.open():
            rec_obj = new_workspace.get_entity(name)[0]
            rec_data = new_workspace.get_entity("DataValues")[0]
            rec_word = new_workspace.get_entity("WordValue")[0]

            compare_entities(points, rec_obj, ignore=["_parent"])
            compare_entities(data, rec_data, ignore=["_parent"])
            compare_entities(word, rec_word, ignore=["_parent"])


def test_create_byte_text_data(tmp_path: Path):
    name = "MyTestPointset"

    # Generate a random cloud of points with reference values
    n_data = 12
    values = np.asarray(
        [
            "".join(random.choice(string.ascii_lowercase) for i in range(8))
            for jj in range(12)
        ]
    )
    values = values.astype(h5py.special_dtype(vlen=str))

    h5file_path = tmp_path / r"testTextData.geoh5"

    with Workspace.create(h5file_path) as workspace:
        with Workspace.create(tmp_path / r"testTextData_copy.geoh5") as new_workspace:
            points = Points.create(
                workspace,
                vertices=np.random.randn(n_data, 3),
                name=name,
                allow_move=False,
            )

            data = points.add_data(
                {
                    "DataValues": {
                        "type": "text",
                        "values": values,
                    }
                }
            )
            word = points.add_data(
                {
                    "Word": {
                        "type": "text",
                        "values": np.array([b"a word"])[0],
                    }
                }
            )

            points.copy(new_workspace)

            assert word.values == np.array([b"a word"])[0].decode("utf-8")
            assert all(data.values == values)

    with workspace.open():
        with new_workspace.open():
            rec_obj = new_workspace.get_entity(name)[0]
            rec_data = new_workspace.get_entity("DataValues")[0]
            rec_word = new_workspace.get_entity("Word")[0]

            compare_entities(points, rec_obj, ignore=["_parent"])
            compare_entities(data, rec_data, ignore=["_parent"])
            compare_entities(word, rec_word, ignore=["_parent"])

            word.values = np.array([b"b word"])[0]
            assert word.values == "b word"


def test_create_one_text_data(tmp_path: Path):
    """
    Would be great to visualize text in GA, if a text object contains 1 text only
    """
    name = "MyTestPointset"

    # Generate a random cloud of points with reference values
    n_data = 12

    h5file_path = tmp_path / r"testTextData.geoh5"

    with Workspace.create(h5file_path) as workspace:
        points = Points.create(
            workspace,
            vertices=np.random.randn(n_data, 3),
            name=name,
            allow_move=False,
        )

        _ = points.add_data(
            {
                "DataValues": {
                    "type": "text",
                    "values": "considering only a text",
                }
            }
        )

    # todo: text cannot be visualize in GA, but could be printed in "DataColour"!
    with Workspace(h5file_path).open("r") as workspace:
        rec_data = workspace.get_entity("DataValues")[0]
        assert "considering only a text" == rec_data.values
