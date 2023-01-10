#  Copyright (c) 2023 Mira Geoscience Ltd Ltd.
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


from __future__ import annotations

import h5py
import pytest

from geoh5py.objects import Points
from geoh5py.shared.exceptions import Geoh5FileClosedError
from geoh5py.workspace import Workspace, active_workspace


def test_workspace_context(tmp_path):
    # TODO: no file on disk should be required for this test
    #       as workspace does not have to be saved
    with active_workspace(Workspace(tmp_path / r"w1.geoh5")) as ws1:
        assert Workspace.active() == ws1
        with active_workspace(Workspace(tmp_path / r"w2.geoh5")) as ws2:
            assert Workspace.active() == ws2
        assert Workspace.active() == ws1
    with pytest.raises(RuntimeError) as error:
        Workspace.active()
    assert "no active workspace" in str(error.value).lower()


def test_write_context(tmp_path):
    with Workspace(tmp_path / r"test2.geoh5", mode="a") as w_s:
        points = Points.create(w_s)

    with pytest.raises(Geoh5FileClosedError) as error:
        w_s.fetch_children(points)

    assert "Consider re-opening" in str(error)

    # Re-open in stanalone readonly
    w_s = Workspace(tmp_path / r"test2.geoh5")
    assert len(w_s.objects) == 1, "Issue creating an object with context manager."
    w_s.close()

    # Re-open in context
    with w_s.open():
        assert isinstance(w_s.geoh5, h5py.File)

    with pytest.raises(Geoh5FileClosedError):
        getattr(w_s, "geoh5")


def test_read_only(tmp_path):
    with pytest.raises(FileNotFoundError) as error:
        with Workspace(tmp_path / r"test.geoh5", mode="r") as w_s:
            Points.create(w_s)

    assert "Unable to open file" in str(error)

    Workspace(tmp_path / r"test.geoh5", mode="a").close()

    with pytest.raises(UserWarning) as error:
        with Workspace(tmp_path / r"test.geoh5", mode="r") as w_s:
            Points.create(w_s)

    assert "geoh5 file in read-only mode" in str(error)
