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

import logging

import numpy as np
import pytest

from geoh5py.objects import Points
from geoh5py.ui_json import UIJson
from geoh5py.ui_json.forms import DataForm, ObjectForm
from geoh5py.ui_json.input_file import DEFAULT_UI_JSON_NAME, InputFile
from geoh5py.workspace import Workspace


class TestUIJson(UIJson):
    title: str = "Test UIJson"
    run_command: str = "run_command"
    conda_environment: str = ""
    points: ObjectForm = ObjectForm(
        label="Object",
        mesh_type="Points",
        value="",
    )
    data: DataForm = DataForm(
        label="Data",
        parent="points",
        data_type="Float",
        association="Vertex",
        value="",
    )


def test_input_file_json(tmp_path, caplog):

    with Workspace.create(tmp_path / f"{__name__}.geoh5") as workspace:
        points = Points.create(
            workspace, vertices=np.random.randn(12, 3), name="Points_A"
        )
        data = points.add_data(
            {
                "values A": {"values": np.random.randn(12)},
                "values B": {"values": np.random.randn(12)},
            }
        )

    uijson = TestUIJson(geoh5=workspace.h5file)

    # Write to file
    uijson.write(tmp_path / "test.ui.json", exclude_unset=False)

    with caplog.at_level(logging.WARNING):
        input_file = InputFile.read_ui_json(tmp_path / "test.ui.json")

    assert "The 'InputFile' class will be deprecated in future" in caplog.text

    # Test setting values
    input_file.update_ui_values({"points": points.uid, "data": data[0].uid})

    # Test getting promoted data
    promoted_data = input_file.data

    assert promoted_data["points"].uid == points.uid

    # Test writing to file
    out_path = input_file.write_ui_json(name="test_input_file.ui.json", path=tmp_path)

    # Read back in and check values
    uijson_read = UIJson.read(out_path)

    assert uijson_read.to_params()["data"].uid == data[0].uid
