import os

import numpy.random as random

from geoh5io.objects import Points
from geoh5io.workspace import Workspace


def test_create_point_data():

    h5file = r"temp\testPoints.geoh5"
    name = "MyTestPointset"
    new_name = "TestName"

    # Generate a random cloud of points
    n_data = 12
    xyz = random.randn(n_data, 3)
    values = random.randn(n_data)

    # Create a workspace
    workspace = Workspace(os.getcwd() + os.sep + "assets" + os.sep + h5file)

    _, data = Points.create(
        workspace,
        vertices=xyz,
        name=name,
        data={"DataValues": ["VERTEX", values]},
        allow_move=False,
    )

    # Change some attributes for testing
    data.allow_delete = False
    data.allow_move = True
    data.allow_rename = False
    data.name = new_name

    workspace.finalize()

    # Read the data back in from a fresh workspace
    workspace = Workspace(os.getcwd() + os.sep + "assets" + os.sep + h5file)

    rec_obj = workspace.get_entity(name)[0]

    rec_data = workspace.get_entity(new_name)[0]

    assert all(rec_data.values == values), "Data values differ from input"
    assert all((rec_obj.locations == xyz).flatten()), "Data locations differ from input"

    check_list = [
        attr
        for attr in rec_data.attribute_map.values()
        if getattr(data, attr) != getattr(rec_data, attr)
    ]
    assert (
        len(check_list) == 0
    ), f"Attribute{check_list} of Data in output differ from input"

    os.remove(os.getcwd() + os.sep + "assets" + os.sep + h5file)
