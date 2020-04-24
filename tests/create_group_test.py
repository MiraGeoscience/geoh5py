import tempfile
from pathlib import Path

from geoh5py.groups import ContainerGroup
from geoh5py.workspace import Workspace


def test_create_group():

    group_name = "MyTestContainer"

    with tempfile.TemporaryDirectory() as tempdir:
        h5file_path = Path(tempdir) / r"testGroup.geoh5"

        # Create a workspace
        workspace = Workspace(h5file_path)

        group = ContainerGroup.create(workspace, name=group_name)
        workspace.save_entity(group)
        workspace.finalize()

        # Read the group back in
        group_object = workspace.get_entity(group_name)

        assert group_object, "Could not read the group object %s" % group_name
