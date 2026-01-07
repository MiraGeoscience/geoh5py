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

import io
from pathlib import Path

import numpy as np
import pytest
from h5py import File

from geoh5py.objects import Points
from geoh5py.shared.exceptions import Geoh5FileClosedError
from geoh5py.shared.utils import compare_entities, fetch_active_workspace
from geoh5py.workspace import Workspace


def test_workspace_from_kwargs(tmp_path: Path):
    h5file_tmp = tmp_path / r"test.geoh5"

    attr = {
        "distance_unit": "feet",
    }

    workspace = Workspace(**attr).save_as(h5file_tmp)  # type: ignore

    # Test re-opening in read-only - stays in r+"
    with pytest.warns(UserWarning) as warning:
        workspace.open(mode="r")

    assert f"Workspace already opened in mode {workspace.geoh5.mode}." in str(
        warning[0]
    )

    workspace.close()

    workspace = Workspace(h5file_tmp)
    for key, value in attr.items():
        if getattr(workspace, key, None) is not None:
            assert getattr(workspace, key.lower()) == value, (
                f"Error changing value for attribute {key}."
            )

    workspace.close()


def test_empty_workspace(tmp_path):
    Workspace.create(
        tmp_path / r"test.geoh5",
    ).close()

    with File(tmp_path / r"test.geoh5", "r+") as file:
        del file["GEOSCIENCE"]["Groups"]
        del file["GEOSCIENCE"]["Data"]
        del file["GEOSCIENCE"]["Objects"]
        del file["GEOSCIENCE"]["Root"]
        del file["GEOSCIENCE"]["Types"]

    Workspace(
        tmp_path / r"test.geoh5",
    ).close()

    with File(tmp_path / r"test.geoh5", "r+") as file:
        assert "Types" in file["GEOSCIENCE"], (
            "Failed to regenerate the geoh5 structure."
        )


def test_missing_type(tmp_path):
    Workspace.create(
        tmp_path / r"test.geoh5",
    ).close()
    with File(tmp_path / r"test.geoh5", "r+") as file:
        for group in file["GEOSCIENCE"]["Groups"].values():
            del group["Type"]

    Workspace(
        tmp_path / r"test.geoh5",
    ).close()


def test_bad_extension(tmp_path):
    with pytest.raises(ValueError) as error:
        Workspace(
            tmp_path / r"test.h5",
        )

    assert "Input 'h5file' file must have a 'geoh5' extension." in str(error)


def test_file_not_found(tmp_path):
    with pytest.warns(UserWarning, match="We will attempt to `save` the file for you"):
        Workspace(
            tmp_path / r"test.geoh5",
        )


def test_read_bytes(tmp_path):
    with pytest.warns(UserWarning, match=""):
        workspace = Workspace(tmp_path / r"test_warning.geoh5")

    assert workspace.h5file.is_file()  # type: ignore

    with Workspace.create(tmp_path / r"test.geoh5") as workspace:
        workspace.create_entity(Points)

    with open(tmp_path / r"test.geoh5", "rb") as in_file:
        byte_data = in_file.read()

    with Workspace(io.BytesIO(byte_data)) as byte_ws:
        byte_objects = byte_ws.objects

    with Workspace(tmp_path / r"test.geoh5") as file_ws:
        file_objects = file_ws.objects

    assert len(byte_objects) == len(file_objects)


def test_reopening_mode(tmp_path):
    with Workspace.create(tmp_path / r"test.geoh5") as workspace:
        pass

    with fetch_active_workspace(workspace, mode="r") as re_open:
        assert re_open.geoh5.mode == "r"

    with workspace.open(mode="r") as workspace:
        with fetch_active_workspace(workspace, mode="r") as re_open:
            assert re_open.geoh5.mode == "r"

        with pytest.warns(UserWarning, match="Closing the workspace in mode 'r'"):
            with fetch_active_workspace(workspace, mode="r+"):
                assert workspace.geoh5.mode == "r+"


def test_in_memory_to_disk():
    workspace = Workspace()
    points = Points.create(workspace, vertices=np.random.randn(12, 3), name="Points_A")

    with Workspace() as out_workspace:
        new_points = points.copy(parent=out_workspace)

    with pytest.raises(Geoh5FileClosedError):
        compare_entities(points, new_points, ignore=["_parent"])

    with out_workspace.open():
        compare_entities(points, new_points, ignore=["_parent"])

    workspace.close()


def test_network_drive_warning(tmp_path):
    network_drive = tmp_path / "Dropbox"
    network_drive.mkdir()
    with pytest.warns(match="Opening workspace with write access in a network drive"):
        _ = Workspace(network_drive / "test.geoh5")
