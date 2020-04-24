import os

import numpy as np
from scipy import spatial

from geoh5py.objects import Points, Surface
from geoh5py.shared import Entity, EntityType
from geoh5py.workspace import Workspace


def test_copy_entity():

    h5file = r"testProject.geoh5"
    new_h5file = r"testProject_2.geoh5"

    # Generate a random cloud of points
    n_data = 12
    xyz = np.random.randn(n_data, 3)
    values = np.random.randn(n_data)

    # Create a workspace
    workspace = Workspace(os.path.join(os.getcwd(), h5file))
    points = Points.create(workspace, vertices=xyz)
    data = points.add_data({"DataValues": {"association": "VERTEX", "values": values}})

    # Create surface
    surf_2d = spatial.Delaunay(xyz[:, :2])

    # Create a geoh5 surface
    surface = Surface.create(
        workspace, name="mySurf", vertices=xyz, cells=getattr(surf_2d, "simplices")
    )

    workspace.finalize()

    # Read the data back in from a fresh workspace
    new_workspace = Workspace(os.path.join(os.getcwd(), new_h5file))

    points.copy(parent=new_workspace)
    surface.copy(parent=new_workspace)

    rec_points = new_workspace.get_entity(points.name)[0]
    rec_surface = new_workspace.get_entity(surface.name)[0]
    rec_data = new_workspace.get_entity(data.name)[0]

    def compare_objects(object_a, object_b):
        for attr in object_a.__dict__.keys():
            if attr in ["_workspace", "_children", "_uid"]:
                continue
            if isinstance(getattr(object_a, attr[1:]), (Entity, EntityType)):
                compare_objects(
                    getattr(object_a, attr[1:]), getattr(object_b, attr[1:])
                )
            else:
                assert np.all(
                    getattr(object_a, attr[1:]) == getattr(object_b, attr[1:])
                ), f"Output attribute {attr[1:]} for {object_a} do not match input {object_b}"

    compare_objects(points, rec_points)
    compare_objects(surface, rec_surface)
    compare_objects(data, rec_data)

    os.remove(os.path.join(os.getcwd(), h5file))
    os.remove(os.path.join(os.getcwd(), new_h5file))
