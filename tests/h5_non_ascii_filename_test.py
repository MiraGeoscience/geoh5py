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

import sys
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import pytest

from geoh5py import Workspace
from geoh5py.ui_json.constants import default_ui_json
from geoh5py.ui_json.input_file import InputFile


NON_ASCII_FILENAME = (
    r"éèçàù¨ẫäêëîïôöûü" + r"ὦშข้าእግᚾᚩᚱᚦᚹ⠎⠁⠹ ⠹ ∀x∈ℝ ٩(-̮̮̃-̃)۶ ٩(●̮̮̃•̃)۶ ٩(͡๏̯͡๏)۶ ٩(-̮̮̃•̃).h5"
)


@pytest.mark.xfail(
    sys.platform == "win32",
    raises=UnicodeEncodeError,
    reason="H5 library version < 1.12 does not support non-ASCII filename",
)
def test_write_reread_non_ascii_filename(tmp_path):
    dataset_name = "mydataset"
    dataset_shape = (100,)

    file_path = tmp_path / NON_ASCII_FILENAME
    # create a new file
    with h5py.File(file_path, "w") as h5_file:
        dataset = h5_file.create_dataset(dataset_name, dataset_shape, dtype="i")
        assert dataset is not None

    # re-open that file
    with h5py.File(file_path, "r") as h5_file:
        dataset = h5_file[dataset_name]
        assert dataset is not None
        assert dataset.shape == dataset_shape
        assert dataset.dtype == np.dtype("int32")


@pytest.mark.xfail(
    sys.platform == "win32",
    raises=UnicodeEncodeError,
    reason="H5 library version < 1.12 does not support non-ASCII filename",
)
def test_existing_non_ascii_filename(tmp_path: Path):
    file_path = tmp_path / NON_ASCII_FILENAME
    with open(file_path, "w", encoding="utf-8"):
        pass

    assert file_path.is_file()
    assert not h5py.is_hdf5(file_path)


def test_non_ascii_path_geoh5(tmp_path: Path):
    path = tmp_path / NON_ASCII_FILENAME
    path.mkdir(parents=True, exist_ok=True)
    workspace = Workspace.create(path / "test.geoh5")
    workspace.close()

    ifile = InputFile(ui_json=deepcopy(default_ui_json))
    ifile.update_ui_values({"geoh5": workspace})
    ifile.write_ui_json("test.ui.json", path)

    # Read file back
    new_read = InputFile.read_ui_json(path / "test.ui.json")

    assert all(
        part_a == part_b
        for part_a, part_b in zip(
            new_read.data["geoh5"].h5file.parts, workspace.h5file.parts, strict=False
        )
    )

    with Workspace(str(path / "test.geoh5")) as workspace:
        assert workspace.h5file == path / "test.geoh5"
