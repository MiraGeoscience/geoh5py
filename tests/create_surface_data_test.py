import os

import numpy as np
from scipy import spatial

from geoh5io.objects import Surface
from geoh5io.workspace import Workspace


def test_create_surface_data():

    h5file = r"testSurface.geoh5"

    workspace = Workspace(os.getcwd() + os.sep + "assets" + os.sep + h5file)

    # Create a grid of points and triangulate
    x, y = np.meshgrid(np.arange(10), np.arange(10))
    x, y = x.ravel(), y.ravel()
    z = np.random.randn(x.shape[0])

    del_surf = spatial.Delaunay(np.c_[x, y])

    simplices = getattr(del_surf, "simplices")

    # Create random data
    values = np.mean(
        np.c_[x[simplices[:, 0]], x[simplices[:, 1]], x[simplices[:, 2]]], axis=1
    )

    # Create a geoh5 surface
    surface = Surface.create(
        workspace, name="mySurf", vertices=np.c_[x, y, z], cells=simplices
    )

    data = surface.add_data({"TMI": {"values": values}})

    # Read the object from a different workspace
    new_workspace = Workspace(
        os.getcwd() + os.sep + "assets" + os.sep + r"testSurface.geoh5"
    )

    obj_copy = new_workspace.get_entity("mySurf")[0]
    data_copy = obj_copy.get_data("TMI")[0]

    assert [
        prop in obj_copy.get_data_list() for prop in surface.get_data_list()
    ], "The surface object did not copy"
    assert np.all(data_copy.values == data.values), "Data values were not copied"

    os.remove(os.getcwd() + os.sep + "assets" + os.sep + r"testSurface.geoh5")
