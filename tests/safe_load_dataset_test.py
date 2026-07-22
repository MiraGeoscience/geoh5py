# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2026 Mira Geoscience Ltd.                                '
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

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from geoh5py.io.h5_reader import safe_load_dataset
from geoh5py.shared.exceptions import MemoryValidationError


def make_mock_dataset(data: np.ndarray) -> MagicMock:
    """Create a mock h5py.Dataset backed by a numpy array."""
    mock = MagicMock()
    mock.size = data.size
    mock.dtype = data.dtype
    mock.__getitem__ = MagicMock(side_effect=lambda s: data[s])
    return mock


def test_safe_load_dataset_success():
    """Normal case: enough memory, data is loaded correctly."""
    data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    mock_ds = make_mock_dataset(data)

    # Estimated bytes: 3 * 8 = 24 bytes — tiny, always fits
    result = safe_load_dataset(mock_ds, "test_key")

    np.testing.assert_array_equal(result, data)


def test_safe_load_dataset_insufficient_memory():
    """Pre-check fails: estimated size exceeds available memory."""
    data = np.ones(1000, dtype=np.float64)
    mock_ds = make_mock_dataset(data)

    # Patch psutil to report very little available memory (1 byte)
    with patch("geoh5py.io.h5_reader.psutil.virtual_memory") as mock_vm:
        mock_vm.return_value.available = 1

        with pytest.raises(MemoryValidationError):
            safe_load_dataset(mock_ds, "test_key")


def test_safe_load_dataset_memory_error_on_load():
    """Dataset passes pre-check but raises MemoryError when sliced."""
    data = np.ones(10, dtype=np.float64)
    mock_ds = MagicMock()
    mock_ds.size = data.size
    mock_ds.dtype = data.dtype
    mock_ds.__getitem__ = MagicMock(side_effect=MemoryError("out of memory"))

    # Give plenty of available memory so pre-check passes
    with patch("geoh5py.io.h5_reader.psutil.virtual_memory") as mock_vm:
        mock_vm.return_value.available = 10 * 1024**3  # 10 GB

        with pytest.raises(MemoryValidationError):
            safe_load_dataset(mock_ds, "test_key")


def test_safe_load_dataset_custom_buffer():
    """Buffer parameter correctly scales the available memory threshold."""
    data = np.ones(1000, dtype=np.float64)
    mock_ds = make_mock_dataset(data)

    with patch("geoh5py.io.h5_reader.psutil.virtual_memory") as mock_vm:
        # Available = 10000 bytes. With buffer=0.5 -> threshold = 5000 < 8000 -> should fail
        mock_vm.return_value.available = 10_000

        with pytest.raises(MemoryValidationError):
            safe_load_dataset(mock_ds, "test_key", buffer=0.5)

        # With buffer=1.0 -> threshold = 10000 >= 8000 -> should succeed
        result = safe_load_dataset(mock_ds, "test_key", buffer=1.0)
        np.testing.assert_array_equal(result, data)


def test_safe_load_dataset_error_message_contains_key():
    """MemoryValidationError message includes the dataset key."""
    data = np.ones(1000, dtype=np.float64)
    mock_ds = make_mock_dataset(data)

    with patch("geoh5py.io.h5_reader.psutil.virtual_memory") as mock_vm:
        mock_vm.return_value.available = 1

        with pytest.raises(MemoryValidationError, match="test_key"):
            safe_load_dataset(mock_ds, "test_key")
