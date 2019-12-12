import os

import numpy.random as random

from geoh5io.objects import Points
from geoh5io.workspace import Workspace


def test_save_data():

    h5file = "Test_float.geoh5"
    name = "MyTestPointset"

    # Generate a random cloud of points
    n_data = 12
    xyz = random.randn(n_data, 3)
    values = random.randn(n_data)

    # value = random.randn(1)

    # Create a workspace
    workspace = Workspace(os.getcwd() + os.sep + "assets" + os.sep + h5file)
    points, data = Points.create(
        workspace,
        vertices=xyz,
        name=name,
        data={"DataValues": ["VERTEX", values]},
        allow_move=False,
    )

    # Add data directly to a new workspace
    new_workspace = Workspace(
        os.getcwd() + os.sep + "assets" + os.sep + "Test_float.geoh5"
    )
    new_workspace.save_entity(data)
    new_workspace.finalize()

    # Get the parent object from the new workspace
    new_points = new_workspace.get_entity(name)[0]
    new_data = new_workspace.get_entity("DataValues")[0]

    assert all(data.values == new_data.values), "Data values differ from input"
    assert all(
        (points.locations == new_points.locations).flatten()
    ), "Data locations differ from input"

    os.remove(os.getcwd() + os.sep + "assets" + os.sep + h5file)
