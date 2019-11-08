import os

from geoh5io.groups import ContainerGroup
from geoh5io.workspace import Workspace, active_workspace


def test_create_group():

    h5file = "testProject.geoh5"

    group_name = "MyContainer"

    # Create a workspace
    workspace = Workspace(r".\assets" + os.sep + h5file)

    with active_workspace(workspace):

        group = ContainerGroup.create(name=group_name)
        group.save_to_h5()
        workspace.finalize()

    # Read the group back in
    group_object = workspace.get_entity(group_name)

    assert group_object, "Could not read the group object %s" % group_name

    os.remove(r".\assets" + os.sep + h5file)
