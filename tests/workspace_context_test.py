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
