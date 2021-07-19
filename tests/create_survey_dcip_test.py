#  Copyright (c) 2021 Mira Geoscience Ltd.
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
# from geoh5py.objects import Curve
from geoh5py.workspace import Workspace

# from geoh5py.shared.utils import compare_entities
# def test_survey_dcip():

NAME = "TestCurve"

# Generate survey lines
N_DATA = 12

# with tempfile.TemporaryDirectory() as tempdir:

# PATH = Path(tempdir) / r"testCurve.geoh5"
PATH = r"C:\Users\dominiquef\Desktop\dcip_work.geoh5"

# Create a workspace
workspace = Workspace(PATH)

currents = workspace.get_entity("Generic - DC/IP (currents")[0]
# curve = Curve.create(workspace, vertices=np.random.randn(N_DATA, 3), name=NAME)

# Get and change the parts
# parts = curve.parts
# parts[-3:] = 1
# curve.parts = parts
#
# data_objects = curve.add_data(
#     {
#         "vertexValues": {"values": np.random.randn(curve.n_vertices)},
#         "cellValues": {"values": np.random.randn(curve.n_cells)},
#     }
# )
#
# workspace.finalize()
# # Re-open the workspace and read data back in
# workspace = Workspace(PATH)
#
# obj_rec = workspace.get_entity(NAME)[0]
# data_vert_rec = workspace.get_entity("vertexValues")[0]
# data_cell_rec = workspace.get_entity("cellValues")[0]
#
# # Check entities
# compare_entities(curve, obj_rec)
# compare_entities(data_objects[0], data_vert_rec)
# compare_entities(data_objects[1], data_cell_rec)
#
# # Modify and write
# obj_rec.vertices = np.random.randn(N_DATA, 3)
# data_vert_rec.values = np.random.randn(N_DATA)
# workspace.finalize()
#
# # Read back and compare
# workspace = Workspace(PATH)
# obj = workspace.get_entity(NAME)[0]
# data_vertex = workspace.get_entity("vertexValues")[0]
#
# compare_entities(obj_rec, obj)
# compare_entities(data_vert_rec, data_vertex)
