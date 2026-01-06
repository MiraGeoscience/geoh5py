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

import h5py
import pytest

from geoh5py.objects import Points
from geoh5py.shared.exceptions import Geoh5FileClosedError
from geoh5py.workspace import Workspace, active_workspace


def test_workspace_context():
    with active_workspace(Workspace()) as ws1:
        assert Workspace.active() == ws1
        with active_workspace(Workspace()) as ws2:
            assert Workspace.active() == ws2
        assert Workspace.active() == ws1
    with pytest.raises(RuntimeError) as error:
        Workspace.active()
    assert "no active workspace" in str(error.value).lower()


def test_write_context():
    with Workspace() as w_s:
        points = Points.create(w_s)

    with pytest.raises(Geoh5FileClosedError):
        w_s.fetch_children(points)

    # Re-open in standalone readonly
    w_s.open()
    assert len(w_s.objects) == 1, "Issue creating an object with context manager."
    w_s.close()

    # Re-open in context
    with w_s.open():
        assert isinstance(w_s.geoh5, h5py.File)

    with pytest.raises(Geoh5FileClosedError):
        w_s.geoh5


def test_read_only():
    with pytest.raises(UserWarning, match="geoh5 file in read-only mode"):
        with Workspace(mode="r") as workspace:
            Points.create(workspace)


def test_deprecation_warnings(tmp_path):
    with pytest.warns(
        match="must be a string or path to an existing file",
    ):
        workspace = Workspace(tmp_path / r"test.geoh5")

    with pytest.warns(
        DeprecationWarning,
        match="The 'finalize' method will be deprecated",
    ):
        workspace.finalize()
