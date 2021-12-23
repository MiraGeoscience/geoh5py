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

from os import path

import pytest

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
