import os

from numpy import c_, cos, linspace, pi, zeros

from geoh5py.objects import Curve
from geoh5py.workspace import Workspace


def test_create_property_group():

    h5file = r"prop_group_test.geoh5"

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
        curve.add_data({f"Period{i+1}": {"values": values}}, property_group="myGroup")

    # Property group object should have been created
    prop_group = curve.get_property_group("myGroup")

    # Create a new group by data name
    single_data_group = curve.add_data_to_group(f"Period{1}", "Singleton")

    assert (
        workspace.find_data(single_data_group.properties[0]).name == f"Period{1}"
    ), "Failed at creating a property group by data name"
    workspace.finalize()

    # Re-open the workspace
    workspace = Workspace(os.getcwd() + os.sep + "assets" + os.sep + h5file)

    # Read the property_group back in
    rec_prop_group = workspace.get_entity(obj_name)[0].get_property_group("myGroup")

    attrs = rec_prop_group.attribute_map
    check_list = [
        attr
        for attr in attrs.values()
        if getattr(rec_prop_group, attr) != getattr(prop_group, attr)
    ]
    assert (
        len(check_list) == 0
    ), f"Attribute{check_list} of PropertyGroups in output differ from input"

    os.remove(os.getcwd() + os.sep + "assets" + os.sep + h5file)
