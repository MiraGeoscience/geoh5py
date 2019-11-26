import os

import numpy.random as random

from geoh5io.objects import Points
from geoh5io.workspace import Workspace


def test_create_point_data():

    h5file = "testProject.geoh5"
    name = "MyTestPointset"

    # Generate a random cloud of points
    n_data = 12
    xyz = random.randn(n_data, 3)
    values = random.randn(n_data)

    # Create a workspace
    workspace = Workspace(r".\assets" + os.sep + h5file)

    points, data = Points.create(
        workspace,
        vertices=xyz,
        name=name,
        data={"DataValues": ["VERTEX", values]},
        allow_move=False,
    )

    assert not points.allow_move, "Attribute of point did not properly set on creation"

    points.save_to_h5()

    workspace.finalize()

    # Read the data back in from a fresh workspace
    workspace = Workspace(r".\assets" + os.sep + h5file)
    obj_list = workspace.list_objects

    obj = workspace.get_entity(obj_list[0])[0]

    data = workspace.get_entity(obj.get_data_list[0])[0]

    assert all(data.values == values), "Data values differ from input"
    assert all((obj.locations == xyz).flatten()), "Data locations differ from input"

    os.remove(r".\assets" + os.sep + h5file)
