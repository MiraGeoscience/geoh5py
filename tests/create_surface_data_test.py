import os

import numpy as np

from geoh5io.workspace import Workspace


def test_create_surface_data():

    h5file = "surfaceObject.geoh5"

    workspace = Workspace(os.getcwd() + os.sep + "assets" + os.sep + h5file)

    obj = workspace.get_entity("mySurf")[0]

    # Create data on cells
    cell_values = np.random.randn(obj.n_cells)

    # Create a new workspace and save object to it
    data_object = obj.add_data({"RandomValues": ["CELL", cell_values]})[0]

    # Write the object to a different workspace
    new_workspace = Workspace(
        os.getcwd() + os.sep + "assets" + os.sep + "testSurface.geoh5"
    )

    new_workspace.save_entity(obj)
    new_workspace.finalize()

    obj_copy = new_workspace.get_entity("mySurf")[0]
    data_copy = obj_copy.get_data(data_object.name)

    assert [
        prop in obj_copy.get_data_list for prop in obj.get_data_list
    ], "The surface object did not copy"
    assert np.all(data_copy.values == data_object.values), "Data values were not copied"
    os.remove(os.getcwd() + os.sep + "assets" + os.sep + "testSurface.geoh5")
