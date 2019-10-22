import numpy as np

from geoh5io.workspace import Workspace


def test_load_data():
    workspace = Workspace(r".\assets\pointObject.geoh5")

    names = workspace.list_objects

    obj = workspace.get_entity(names[0])[0]

    children = obj.get_data_list

    data = obj.get_data(children[0])

    assert isinstance(data.values, np.ndarray), "Imported data should be a numpy array"
