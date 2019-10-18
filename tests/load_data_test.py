from geoh5io.workspace import Workspace


def test_load_data():
    workspace = Workspace(
        r"C:\Users\DominiqueFournier\Dropbox\Projects\Mira\GeoIO\"TKC.geoh5"
    )

    names = workspace.list_objects

    obj = workspace.get_entity(names[0])[0]

    children = obj.get_data_list

    data = obj.get_data(children[0])

    print(data)
