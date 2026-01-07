# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2020-2026 Mira Geoscience Ltd.                                '
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

from pathlib import Path

import numpy as np
import pytest

from geoh5py.data import (
    DataAssociationEnum,
    FloatData,
    IntegerData,
    PrimitiveTypeEnum,
    ReferencedData,
)
from geoh5py.data.data_type import DataType
from geoh5py.objects import VPModel
from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace


# pylint: disable=too-many-locals


def create_mesh_parameters():
    """
    Utility function to generate basic drape model
    """
    n_col, n_row = 64, 32
    j, i = np.meshgrid(np.arange(n_row), np.arange(n_col))

    n_cells = n_row * n_col
    top = np.random.randn(n_cells) * 10.0 + 100.0

    indices = ((i * n_row + j).T * 2).flatten()
    prisms = np.c_[top, indices, np.ones(n_cells, dtype=int) * 2]
    layers = np.c_[
        np.kron(i.flatten(), np.ones(2)),
        np.kron(j.flatten(), np.ones(2)),
        np.kron(np.ones(n_cells, dtype=int), np.r_[0, -1000]),
    ]
    units = (layers[:, 2] < 0).astype(int) * 99999
    units += 1
    return layers, prisms, units


def test_create_vp_model(tmp_path: Path):
    h5file_path = tmp_path / f"{__name__}.geoh5"
    with Workspace.create(h5file_path) as workspace:
        with pytest.raises(TypeError, match="Attribute 'u_count' "):
            VPModel.validate_count("abc", "u")

        layers, prisms, units = create_mesh_parameters()

        drape = VPModel.create(
            workspace,
            u_count=64,
            u_cell_size=25,
            v_count=32,
            v_cell_size=25,
            layers=layers,
            prisms=prisms,
            unit_property_id=units,
        )

        assert drape.n_cells == layers.shape[0]
        assert drape.centroids.shape == (drape.n_cells, 3)

    with pytest.raises(TypeError, match="Attribute 'flag_property_id' should be"):
        drape.flag_property_id = "abc"

    with pytest.raises(
        TypeError, match="Attribute 'heterogeneous_property_id' should be"
    ):
        drape.heterogeneous_property_id = "abc"

    with pytest.raises(TypeError, match="Attribute 'physical_data_name' should be"):
        drape.physical_data_name = 1.0

    with pytest.raises(TypeError, match="Attribute 'weight_property_id' should be"):
        drape.weight_property_id = "abc"

    ws = Workspace()

    with pytest.raises(
        TypeError, match="Attribute 'unit_property_id' should be a 'uuid.UUID'"
    ):
        VPModel.create(ws, layers=layers, prisms=prisms, unit_property_id="abc")

    assert isinstance(drape.visual_parameters.filter_basement, float)
    np.testing.assert_almost_equal(
        drape.visual_parameters.filter_basement,
        (prisms[:, 0].max() - layers[:, 2].min()) * 0.05,
    )


@pytest.mark.parametrize(
    ("name", "dtype"),
    [
        ("flag_property_id", IntegerData),
        ("heterogeneous_property_id", FloatData),
        ("unit_property_id", ReferencedData),
        ("weight_property_id", FloatData),
    ],
)
def test_vp_valiations(name, dtype: DataType, tmp_path: Path):
    with Workspace.create(tmp_path / f"{__name__}.geoh5") as workspace:
        layers, prisms, units = create_mesh_parameters()

        entity_type = DataType.find_or_create_type(workspace, PrimitiveTypeEnum(dtype))

        kwargs = {
            name: dtype(association=DataAssociationEnum.OBJECT, entity_type=entity_type)
        }
        with pytest.raises(ValueError, match="not a child of the VPModel"):
            VPModel.create(
                workspace,
                u_count=64,
                u_cell_size=25,
                v_count=32,
                v_cell_size=25,
                layers=layers,
                prisms=prisms,
                **kwargs,
            )

        vp = VPModel.create(
            workspace,
            u_count=64,
            u_cell_size=25,
            v_count=32,
            v_cell_size=25,
            layers=layers,
            prisms=prisms,
        )

        with pytest.raises(
            TypeError, match=f"Attribute '{name}' should be a 'uuid.UUID'."
        ):
            setattr(vp, name, "abc")

        setattr(vp, name, kwargs[name])


def test_modify_vp_model(tmp_path: Path):
    h5file_path = tmp_path / f"{__name__}.geoh5"
    with Workspace.create(h5file_path) as workspace:
        layers, prisms, units = create_mesh_parameters()

        drape = VPModel.create(
            workspace,
            u_count=64,
            u_cell_size=25,
            v_count=32,
            v_cell_size=25,
            layers=layers,
            prisms=prisms,
            unit_property_id=units,
        )
        drape.u_cell_size = 50
        drape.v_cell_size = 50

    with workspace.open():
        vp = workspace.get_entity(drape.uid)[0]
        with Workspace.create(tmp_path / f"{__name__}_copy.geoh5") as new_ws:
            vp_copy = vp.copy(parent=new_ws)

            np.testing.assert_almost_equal(vp_copy.u_cell_size, 50)
