#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoh5py.
#
#  geoh5py is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  geoh5py is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with geoh5py.  If not, see <https://www.gnu.org/licenses/>.


# import numpy as np
#
# from geoh5py.objects import Grid2D
# from geoh5py.shared.utils import compare_entities
# from geoh5py.workspace import Workspace


def test_create_drape_model():
    pass
    # name = "MyDrapeModel"
    # h5file_path = os.path.join(
    #     r"C:\Users\dominiquef\Documents\GIT\mira\geoh5py\tests", "A LINE.geoh5"
    # )
    # workspace = Workspace(h5file_path)
    #
    # drape_model = workspace.get_entity("A LINE02__")[0]
    # new_workspace = Workspace("tester.geoh5")
    # drape_model.copy(parent=new_workspace)
    #
    # new_drape = new_workspace.get_entity("A LINE02__")[0]
    # compare_entities(drape_model, new_drape, ignore=["_parent"])
    # Generate a 2D array
    # n_x, n_y = 10, 15
    # values, _ = np.meshgrid(np.linspace(0, np.pi, n_x), np.linspace(0, np.pi, n_y))
    #
    # with tempfile.TemporaryDirectory() as tempdir:
    #     h5file_path = Path(tempdir) / r"test2Grid.geoh5"
    #
    #     # Create a workspace
    #     workspace = Workspace(h5file_path)
    #
    #     grid = Grid2D.create(
    #         workspace,
    #         origin=[0, 0, 0],
    #         u_cell_size=20.0,
    #         v_cell_size=30.0,
    #         u_count=n_x,
    #         v_count=n_y,
    #         name=name,
    #         allow_move=False,
    #     )
    #
    #     data = grid.add_data({"DataValues": {"values": values}})
    #     grid.rotation = 45.0
    #
    #     workspace.finalize()
    #
    #     # Read the data back in from a fresh workspace
    #     new_workspace = Workspace(h5file_path)
    #
    #     rec_obj = new_workspace.get_entity(name)[0]
    #
    #     rec_data = new_workspace.get_entity("DataValues")[0]
    #
    #     compare_entities(grid, rec_obj)
    #     compare_entities(data, rec_data)
