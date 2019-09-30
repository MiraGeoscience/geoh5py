import pytest

from geoh5io.workspace import Workspace, active_workspace


def test_workspace_context():
    with active_workspace(Workspace()) as ws1:
        assert Workspace.active() == ws1
        with active_workspace(Workspace()) as ws2:
            assert Workspace.active() == ws2
        assert Workspace.active() == ws1
    with pytest.raises(RuntimeError) as error:
        Workspace.active()
    assert "no active workspace" in str(error).lower()
