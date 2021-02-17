#  Copyright (c) 2020 Mira Geoscience Ltd.
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

import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

NON_ASCII_FILENAME = (
    r"éèçàù¨ẫäêëîïôöûü"
    + r"ὦშข้าእግᚾᚩᚱᚦᚹ⠎⠁⠹ ⠹ ∀x∈ℝ ٩(-̮̮̃-̃)۶ ٩(●̮̮̃•̃)۶ ٩(͡๏̯͡๏)۶ ٩(-̮̮̃•̃).h5"
)


@pytest.mark.xfail(
    sys.platform == "win32",
    raises=UnicodeEncodeError,
    reason="H5 library version < 1.12 does not support non-ASCII filename",
)
def test_write_reread_non_ascii_filename():
    with tempfile.TemporaryDirectory() as tempdir:
        dataset_name = "mydataset"
        dataset_shape = (100,)

        file_path = Path(tempdir) / NON_ASCII_FILENAME
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
def test_existing_non_ascii_filename():
    with tempfile.TemporaryDirectory() as tempdir:
        file_path = Path(tempdir) / NON_ASCII_FILENAME
        with open(file_path, "w"):
            pass

        assert file_path.exists()
        assert not h5py.is_hdf5(file_path)
