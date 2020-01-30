import os

from numpy import c_, cos, linspace, pi, zeros

from geoh5io.objects import Curve
from geoh5io.workspace import Workspace


def test_create_property_group():

    h5file = r"temp\prop_group_test.geoh5"

    obj_name = "myCurve"
    # Generate a curve with multiple data
    n_stn = 12
    xyz = c_[linspace(0, 2 * pi, n_stn), zeros(n_stn), zeros(n_stn)]

    # Create a workspace
    workspace = Workspace(os.getcwd() + os.sep + "assets" + os.sep + h5file)

    curve = Curve.create(workspace, vertices=xyz, name=obj_name)

    # Add data
    for i in range(4):
        values = cos(xyz[:, 0] / (i + 1))
        curve.add_data({f"Period{i+1}": ["VERTEX", values]}, property_group="myGroup")

    # Property group object should have been created
    prop_group = curve.get_property_group("myGroup")

    workspace.save_entity(curve)
    workspace.finalize()

    # Re-open the workspace
    workspace = Workspace(os.getcwd() + os.sep + "assets" + os.sep + h5file)

    # Read the property_group back in
    rec_prop_group = workspace.get_entity(obj_name)[0].property_groups[0]

    attrs = rec_prop_group.__dict__
    check_list = [
        attr
        for attr in attrs.keys()
        if getattr(rec_prop_group, attr) != getattr(prop_group, attr)
    ]
    assert (
        len(check_list) == 0
    ), f"Attribute{check_list} of PropertyGroups in output differ from input"

    os.remove(os.getcwd() + os.sep + "assets" + os.sep + h5file)
