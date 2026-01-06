# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2025-2026 Mira Geoscience Ltd.                                '
#                                                                              '
#  This file is part of geoh5py.                                               '
#                                                                              '
#  geoh5py is free software: you can redistribute it and/or modify             '
#  it under the terms of the GNU Lesser General Public License as published by '
#  the Free Software Foundation, either version 3 of the License, or           '
#  (at your option) any later version.                                         '
#                                                                              '
#  geoh5py is distributed in the hope that it will be useful,                  '
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              '
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               '
#  GNU Lesser General Public License for more details.                         '
#                                                                              '
#  You should have received a copy of the GNU Lesser General Public License    '
#  along with geoh5py.  If not, see <https://www.gnu.org/licenses/>.           '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pytest

from geoh5py.objects import Curve
from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace


def test_attribute_validations():
    # Generate a random cloud of points
    n_data = 12

    with Workspace() as workspace:
        with pytest.raises(TypeError, match="Parts must be a list or numpy array."):
            Curve.create(workspace, vertices=np.random.randn(n_data, 3), parts="abc")

        with pytest.raises(
            TypeError, match="Attribute 'cells' must be provided as type"
        ):
            Curve.create(workspace, vertices=np.random.randn(n_data, 3), cells="abc")

        with pytest.raises(TypeError, match="Indices array must be of integer type"):
            Curve.create(
                workspace,
                vertices=np.random.randn(n_data, 3),
                cells=np.c_[np.arange(n_data - 1), np.arange(n_data - 1)].astype(float),
            )

        with pytest.raises(ValueError, match="Provided parts must be of shape"):
            Curve.create(
                workspace,
                vertices=np.random.randn(n_data, 3),
                parts=np.ones(n_data - 1),
            )

        with pytest.raises(ValueError, match="Found cell indices larger than"):
            Curve.create(
                workspace,
                vertices=np.random.randn(n_data, 3),
                cells=np.c_[np.arange(11), np.arange(2, 13)],
            )

        curve = Curve.create(workspace, vertices=np.random.randn(n_data, 3))

        with pytest.raises(
            TypeError, match="Input current_line_id value should be of type"
        ):
            curve.current_line_id = "abc"

        new_value = uuid.uuid4()
        curve.current_line_id = new_value

        assert curve.current_line_id == new_value


def test_create_curve_data(tmp_path: Path):
    curve_name = "TestCurve"
    h5file_path = tmp_path / r"testCurve.geoh5"
    # Generate a random cloud of points
    n_data = 12

    with Workspace.create(h5file_path) as workspace:
        curve = Curve.create(workspace, vertices=(1.0, 1.0, 1.0))

        assert curve.vertices.shape == (
            2,
            3,
        ), "Error creating curve with single vertex."
        assert len(curve.cells) == 1
        curve = Curve.create(
            workspace, vertices=np.arange(n_data * 3).reshape(n_data, 3)
        )

        # Get and change the parts
        parts = curve.parts
        parts[-3:] = 1
        with pytest.raises(AttributeError):
            curve.parts = parts

        curve.add_data({"cell_values": {"values": np.ones(curve.n_cells)}})
        cells = curve.cells.copy()

        assert cells.shape[0] == 11, "Error creating cells from parts."

        with pytest.raises(
            ValueError, match="New cells array must have the same shape"
        ):
            curve.cells = np.c_[1, 2]

        np.array_equal(
            np.arange(1, n_data * 3 - 2).reshape(n_data - 1, 3) + 0.5, curve.centroids
        )

        curve = Curve.create(
            workspace, vertices=np.random.randn(n_data, 3), name=curve_name, cells=cells
        )

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

        assert np.all(data_objects[0]() == data_objects[0].values), (
            "Error using the data.call()."
        )
        # Re-open the workspace and read data back in
        with Workspace(h5file_path) as ws2:
            obj_rec = ws2.get_entity(curve_name)[0]
            data_vert_rec = ws2.get_entity("vertexValues")[0]

            # Check entities
            compare_entities(curve, obj_rec)
            compare_entities(data_objects[0], data_vert_rec)
            compare_entities(data_objects[1], ws2.get_entity("cellValues")[0])

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
        curve = Curve.create(
            workspace, name="new_curve", vertices=np.random.randn(n_data, 3)
        )
        data = curve.add_data(
            {
                "cellValues": {
                    "values": np.random.randn(curve.n_cells).astype(np.float64)
                },
            }
        )

        del curve, data

    with Workspace(tmp_path / r"testCurve.geoh5") as workspace:
        curve = workspace.get_entity("new_curve")[0]

        with pytest.raises(
            ValueError, match="Found indices larger than the number of cells."
        ):
            curve.remove_cells([12])

        with pytest.raises(TypeError, match="Indices must be a list or numpy array."):
            curve.remove_cells("abc")

        with pytest.raises(TypeError, match="Indices must be a list or numpy array."):
            curve.remove_vertices("abc")

        curve.remove_cells([0])

        data = curve.get_data("cellValues")[0]

        assert len(data.values) == 10, "Error removing data values with cells."

        curve.remove_cells(np.arange(curve.n_cells))

        assert data.values is None


def test_remove_vertex_data(tmp_path):
    # Generate a random cloud of points
    n_data = 12

    with Workspace.create(tmp_path / r"testCurve.geoh5") as workspace:
        curve = Curve.create(workspace, vertices=np.random.randn(n_data, 3))
        data = curve.add_data(
            {
                "cellValues": {
                    "values": np.random.randn(curve.n_cells).astype(np.float64)
                },
                "vertValues": {
                    "values": np.random.randn(curve.n_vertices).astype(np.float64)
                },
            }
        )

        curve.copy(name="validation")

        with pytest.raises(
            ValueError, match="Found indices larger than the number of vertices."
        ):
            curve.remove_vertices([12])

        curve.remove_vertices([0, 3], clear_cache=True)

        assert len(data[0].values) == 8, "Error removing data values with cells."
        assert len(curve.vertices) == 10, "Error removing vertices from cells."
        with pytest.raises(ValueError, match="Operation would leave fewer"):
            curve.remove_vertices(np.arange(curve.n_vertices))

        curve.remove_vertices([6])
        assert len(np.unique(curve.parts)) == 3, "Error detecting parts."


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

        with pytest.raises(ValueError, match="Mask must be an array of dtype"):
            curve.copy(mask=np.r_[1, 2, 3])

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


def test_cell_from_part(tmp_path):
    n_vertices = 1000
    locations = np.random.randn(n_vertices, 3)
    parts = np.random.randint(0, 10, n_vertices)

    with Workspace.create(tmp_path / r"testCellParts.geoh5") as workspace:
        curve = Curve.create(workspace, vertices=locations, parts=parts)
        curve.add_data(
            {
                "parts": {"values": parts},
            }
        )
        cells = curve.cells

    end_points = cells[:-1, 1] != cells[1:, 0]
    assert np.sum(end_points) == 9, "Error creating cells from parts."

    ascending = cells[1:, 0] > cells[:-1, 0]
    ascending[end_points] = True

    assert np.all(ascending), "Error creating sorted cells from parts."
