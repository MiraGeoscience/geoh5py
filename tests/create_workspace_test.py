#  Copyright (c) 2022 Mira Geoscience Ltd.
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

from os import path

import pytest
from h5py import File

from geoh5py.workspace import Workspace


def test_workspace_from_kwargs(tmp_path):

    attr = {
        "Contributors": "TARS",
        "version": 999.1,
        "ga_version": "2",
        "distance_unit": "feet",
        "hello": "world",
    }

    with pytest.warns(UserWarning) as warning:
        workspace = Workspace(path.join(tmp_path, "test.geoh5"), **attr)

    assert (
        "UserWarning('Argument hello with value world is not a valid attribute"
        in str(warning[0])
    )
    workspace.finalize()

    workspace = Workspace(
        path.join(tmp_path, "test.geoh5"),
    )
    for key, value in attr.items():
        if getattr(workspace, key, None) is not None:
            assert (
                getattr(workspace, key.lower()) == value
            ), f"Error changing value for attribute {key}."


def test_empty_workspace(tmp_path):
    Workspace(
        path.join(tmp_path, "test.geoh5"),
    )
    with File(path.join(tmp_path, "test.geoh5"), "r+") as file:
        del file["GEOSCIENCE"]["Groups"]
        del file["GEOSCIENCE"]["Data"]
        del file["GEOSCIENCE"]["Objects"]
        del file["GEOSCIENCE"]["Root"]
        del file["GEOSCIENCE"]["Types"]

    Workspace(
        path.join(tmp_path, "test.geoh5"),
    )

    with File(path.join(tmp_path, "test.geoh5"), "r+") as file:
        assert (
            "Types" in file["GEOSCIENCE"]
        ), "Failed to regenerate the geoh5 structure."


def test_missing_type(tmp_path):
    Workspace(
        path.join(tmp_path, "test.geoh5"),
    )
    with File(path.join(tmp_path, "test.geoh5"), "r+") as file:
        for group in file["GEOSCIENCE"]["Groups"].values():
            del group["Type"]

    Workspace(
        path.join(tmp_path, "test.geoh5"),
    )

    with File(path.join(tmp_path, "test.geoh5"), "r+") as file:
        assert all(
            "Type" in group for group in file["GEOSCIENCE"]["Groups"].values()
        ), "Failed to regenerate the Type in geoh5 structure."
