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
    h5file_tmp = path.join(tmp_path, "test.geoh5")

    attr = {
        "Contributors": "TARS",
        "version": 999.1,
        "ga_version": "2",
        "distance_unit": "feet",
        "hello": "world",
    }

    with pytest.warns(UserWarning) as warning:
        workspace = Workspace(h5file_tmp, **attr)

    assert (
        "UserWarning('Argument hello with value world is not a valid attribute"
        in str(warning[0])
    )
    workspace.close()

    workspace = Workspace(h5file_tmp)
    for key, value in attr.items():
        if getattr(workspace, key, None) is not None:
            assert (
                getattr(workspace, key.lower()) == value
            ), f"Error changing value for attribute {key}."
    workspace.close()

    # Add .lock to simulate ANALYST session
    with open(workspace.h5file + ".lock", "a", encoding="utf-8") as lock_file:
        lock_file.write("Hello World")

    workspace = Workspace(h5file_tmp)
    workspace.version = 2.0

    with pytest.warns(UserWarning) as warning:
        workspace.close()

    assert "*.lock" in str(warning[0])


def test_empty_workspace(tmp_path):
    Workspace(
        path.join(tmp_path, "test.geoh5"),
    ).close()

    with File(path.join(tmp_path, "test.geoh5"), "r+") as file:
        del file["GEOSCIENCE"]["Groups"]
        del file["GEOSCIENCE"]["Data"]
        del file["GEOSCIENCE"]["Objects"]
        del file["GEOSCIENCE"]["Root"]
        del file["GEOSCIENCE"]["Types"]

    Workspace(
        path.join(tmp_path, "test.geoh5"),
    ).close()

    with File(path.join(tmp_path, "test.geoh5"), "r+") as file:
        assert (
            "Types" in file["GEOSCIENCE"]
        ), "Failed to regenerate the geoh5 structure."


def test_missing_type(tmp_path):
    Workspace(
        path.join(tmp_path, "test.geoh5"),
    ).close()
    with File(path.join(tmp_path, "test.geoh5"), "r+") as file:
        for group in file["GEOSCIENCE"]["Groups"].values():
            del group["Type"]

    Workspace(
        path.join(tmp_path, "test.geoh5"),
    ).close()


def test_bad_extension(tmp_path):
    with pytest.raises(ValueError) as error:
        Workspace(
            path.join(tmp_path, "test.h5"),
        )

    assert "Input 'h5file' file must have a 'geoh5' extension." in str(error)
