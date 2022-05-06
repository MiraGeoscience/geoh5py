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

from geoh5py.objects import Points
from geoh5py.workspace import Workspace, active_workspace


def test_workspace_context(tmp_path):
    # TODO: no file on disk should be required for this test
    #       as workspace does not have to be saved
    with active_workspace(Workspace(path.join(tmp_path, "w1.geoh5"))) as ws1:
        assert Workspace.active() == ws1
        with active_workspace(Workspace(path.join(tmp_path, "w2.geoh5"))) as ws2:
            assert Workspace.active() == ws2
        assert Workspace.active() == ws1
    with pytest.raises(RuntimeError) as error:
        Workspace.active()
    assert "no active workspace" in str(error.value).lower()


def test_write_context(tmp_path):
    with Workspace(path.join(tmp_path, "test2.geoh5"), mode="a") as w_s:
        Points.create(w_s)

    # Re-open in stanalone readonly
    w_s = Workspace(path.join(tmp_path, "test2.geoh5"))
    assert len(w_s.objects) == 1, "Issue creating an object with context manager."
