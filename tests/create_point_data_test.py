import os

import numpy.random as random

from geoh5io.objects import Points
from geoh5io.workspace import Workspace, active_workspace


def test_create_group():

    h5file = "testProject.geoh5"

    # Generate a random cloud of points
    n_data = 12
    xyz = random.randn(n_data, 3)
    values = random.randn(n_data)

    # Create a workspace
    workspace = Workspace(r".\assets" + os.sep + h5file)

    with active_workspace(workspace):

        points = Points.create(locations=xyz, data={"DataValues": values})
        points.save_to_h5()
        workspace.finalize()

    # Read the data back in
    obj_list = workspace.list_objects

    obj = workspace.get_entity(obj_list[0])[0]

    data = workspace.get_entity(obj.get_data_list[0])[0]

    assert all(data.values == values), "Data values differ from input"

    os.remove(r".\assets" + os.sep + h5file)
