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

from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from geoh5py.groups import ContainerGroup
from geoh5py.objects import Curve
from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace


def test_add_file(tmp_path: Path):
    workspace = Workspace(tmp_path / "test.geoh5")
    workspace_copy = Workspace()
    with pytest.warns(UserWarning, match="No 'vertices' provided."):
        curve = Curve.create(workspace)
    group = ContainerGroup.create(workspace)

    xyz = np.random.randn(32)
    np.savetxt(tmp_path / r"numpy_array.txt", xyz)
    file_name = "numpy_array.txt"
    for obj in [curve, group]:
        file_data = obj.add_file(tmp_path / file_name)
        assert file_data.values == file_name, "File_name not properly set."
        assert file_data.n_values == 1, "Object association should have 1 value."
        # Rename the file locally and write back out
        new_path = tmp_path / r"temp"
        file_data.save_file(path=new_path, name="numpy_array.dat")
        assert (new_path / "numpy_array.dat").is_file(), (
            f"Input path '{new_path / 'numpy_array.dat'}' does not exist."
        )

        file_data.save_file(path=new_path)
        np.testing.assert_array_equal(
            np.loadtxt(new_path / "numpy_array.txt"),
            np.loadtxt(BytesIO(file_data.file_bytes)),
            err_msg="Loaded and stored bytes array not the same",
        )
        file_data.file_bytes = b"abc"
        obj.copy(parent=workspace_copy)
        workspace_copy.close()
        workspace_copy.open()
        copied_obj = workspace_copy.get_entity(obj.uid)[0]
        rec_data = copied_obj.get_entity("numpy_array.txt")[0]
        compare_entities(file_data, rec_data, ignore=["_parent"])

    with pytest.raises(
        TypeError, match="Input 'file_bytes' for FilenameData must be of type 'bytes'."
    ):
        file_data.file_bytes = "abc"

    file_data.values = None

    with pytest.raises(
        AttributeError, match="FilenameData requires the 'values' to be set."
    ):
        file_data.file_bytes = b"abc"


def test_add_file_increment_names(tmp_path: Path):
    """
    Test that adding files with the same name increments the name.
    """
    workspace = Workspace(tmp_path / "test.geoh5")
    group = ContainerGroup.create(workspace)

    file_name = "test.txt"
    names = []
    for _ in range(3):
        xyz = np.random.randn(32)
        np.savetxt(tmp_path / file_name, xyz)
        file_data = group.add_file(tmp_path / file_name)
        names.append(file_data.values)

    assert names == ["test.txt", "test(1).txt", "test(2).txt"]
