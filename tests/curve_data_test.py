#  Copyright (c) 2023 Mira Geoscience Ltd.
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


from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from geoh5py.objects import Curve
from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace


def test_create_curve_data(tmp_path: Path):
    curve_name = "TestCurve"
    h5file_path = tmp_path / r"testCurve.geoh5"
    # Generate a random cloud of points
    n_data = 12

    with Workspace.create(h5file_path) as workspace:
        curve = Curve.create(
            workspace, vertices=np.random.randn(n_data, 3), name=curve_name
        )

        with pytest.raises(
            TypeError, match="Input current_line_id value should be of type"
        ):
            curve.current_line_id = "abc"

        setattr(curve, "_cells", None)
        with pytest.warns(UserWarning, match="No cells to be removed."):
            curve.remove_cells(0)

        # Get and change the parts
        parts = curve.parts
        parts[-3:] = 1
        curve.parts = parts

        cells = curve.cells.copy()
        assert cells.shape[0] == 10, "Error creating cells from parts." ""
        setattr(curve, "_cells", None)
        with pytest.raises(ValueError, match="Array of cells should be of shape"):
            curve.cells = np.c_[1]

        with pytest.raises(TypeError, match="Indices array must be of integer type"):
            curve.cells = np.c_[0.0, 1.0]

        print(curve.parts)

        curve.cells = cells.tolist()

        data_objects = curve.add_data(
            {
                "vertexValues": {
                    "values": np.random.randint(
                        0, curve.n_vertices, curve.n_vertices
                    ).astype(np.uint32)
                },
                "cellValues": {
                    "values": np.random.randn(curve.n_cells).astype(np.float64)
                },
            }
        )

        assert np.all(
            data_objects[0]() == data_objects[0].values
        ), "Error using the data.call()."
        # Re-open the workspace and read data back in
        with Workspace(h5file_path) as ws2:
            obj_rec = ws2.get_entity(curve_name)[0]
            data_vert_rec = ws2.get_entity("vertexValues")[0]

            # Check entities
            compare_entities(curve, obj_rec)
            compare_entities(data_objects[0], data_vert_rec)
            compare_entities(data_objects[1], ws2.get_entity("cellValues")[0])

            # Modify and write
            obj_rec.vertices = np.random.randn(n_data, 3)

            with pytest.raises(TypeError, match="Values cannot have decimal points."):
                data_vert_rec.values = np.random.randn(n_data)  # warning here
            data_vert_rec.values = np.random.randint(
                0, curve.n_vertices, curve.n_vertices
            ).astype(np.uint32)

        # Read back and compare
        with ws2.open():
            with Workspace(h5file_path) as ws3:
                obj = ws3.get_entity(curve_name)[0]
                data_vertex = ws3.get_entity("vertexValues")[0]

                compare_entities(obj_rec, obj)
                compare_entities(data_vert_rec, data_vertex)


def test_remove_cells_data(tmp_path: Path):
    # Generate a random cloud of points
    n_data = 12

    with Workspace.create(tmp_path / r"testCurve.geoh5") as workspace:
        curve = Curve.create(workspace, vertices=np.random.randn(n_data, 3))
        data = curve.add_data(
            {
                "cellValues": {
                    "values": np.random.randn(curve.n_cells).astype(np.float64)
                },
            }
        )

        with pytest.raises(
            ValueError, match="Found indices larger than the number of cells."
        ):
            curve.remove_cells([12])

        with pytest.raises(
            ValueError, match="Attempting to assign 'cells' with fewer values."
        ):
            curve.cells = curve.cells[1:, :]

        with pytest.raises(TypeError, match="Indices must be a list or numpy array."):
            curve.remove_cells("abc")

        with pytest.raises(TypeError, match="Indices must be a list or numpy array."):
            curve.remove_vertices("abc")

        curve.remove_cells([0])

        assert len(data.values) == 10, "Error removing data values with cells."


def test_remove_vertex_data(tmp_path):
    # Generate a random cloud of points
    n_data = 12

    with Workspace.create(tmp_path / r"testCurve.geoh5") as workspace:
        curve = Curve.create(workspace)
        with pytest.warns(UserWarning, match="No vertices to be removed."):
            curve.remove_vertices(12)

        curve.vertices = np.random.randn(n_data, 3)
        data = curve.add_data(
            {
                "cellValues": {
                    "values": np.random.randn(curve.n_cells).astype(np.float64)
                },
            }
        )

        with pytest.raises(
            ValueError, match="Found indices larger than the number of vertices."
        ):
            curve.remove_vertices([12])

        curve.remove_vertices([0, 3])

        assert len(data.values) == 8, "Error removing data values with cells."
        assert len(curve.vertices) == 10, "Error removing vertices from cells."


def test_copy_cells_data(tmp_path):
    # Generate a random cloud of points
    n_data = 12

    with Workspace.create(tmp_path / r"testCurve.geoh5") as workspace:
        curve = Curve.create(workspace, vertices=np.random.randn(n_data, 3))
        data = curve.add_data(
            {
                "cellValues": {
                    "values": np.random.randn(curve.n_cells).astype(np.float64)
                },
            }
        )

        with pytest.raises(ValueError, match="Mask must be an array of shape."):
            curve.copy(mask=[1, 2, 3])

        mask = np.zeros(11, dtype=bool)
        mask[:4] = True
        copy_data = data.copy(mask=mask)

        assert np.isnan(copy_data.values).sum() == 7, "Error copying data."

        ind_vert = np.all(curve.vertices[:, :2] > 0, axis=1) & np.all(
            curve.vertices[:, :2] < 2, axis=1
        )
        ind = np.all(ind_vert[curve.cells], axis=1)
        mask = data.mask_by_extent(np.vstack([[0, 0], [2, 2]]))

        assert np.all(mask == ind), "Error masking cell data by extent."
