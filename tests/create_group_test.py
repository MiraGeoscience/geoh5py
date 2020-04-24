import os

from geoh5py.groups import ContainerGroup
from geoh5py.workspace import Workspace


def test_create_group():

    h5file = r"testGroup.geoh5"

    group_name = "MyTestContainer"

    # Create a workspace
    workspace = Workspace(os.path.join(os.getcwd(), h5file))

    group = ContainerGroup.create(workspace, name=group_name)
    workspace.save_entity(group)
    workspace.finalize()

    # Read the group back in
    group_object = workspace.get_entity(group_name)

    assert group_object, "Could not read the group object %s" % group_name

    os.remove(os.path.join(os.getcwd(), h5file))
