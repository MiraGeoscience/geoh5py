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

import tempfile
from pathlib import Path

import pytest

from geoh5py.workspace import Workspace, active_workspace


def test_workspace_context():
    # TODO: no file on disk should be required for this test
    #       as workspace does not have to be saved
    with tempfile.TemporaryDirectory() as tempdir:
        with active_workspace(Workspace(Path(tempdir) / "w1.geoh5")) as ws1:
            assert Workspace.active() == ws1
            with active_workspace(Workspace(Path(tempdir) / "w2.geoh5")) as ws2:
                assert Workspace.active() == ws2
            assert Workspace.active() == ws1
        with pytest.raises(RuntimeError) as error:
            Workspace.active()
        assert "no active workspace" in str(error.value).lower()
