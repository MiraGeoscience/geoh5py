#  Copyright (c) 2024 Mira Geoscience Ltd.
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

# pylint: disable=R0914


from __future__ import annotations

from pathlib import Path

import numpy as np

from geoh5py.objects import CurrentElectrode, PotentialElectrode
from geoh5py.workspace import Workspace


def test_copy_survey_dcip(tmp_path: Path):
    name = "TestCurrents"
    n_data = 12
    path = tmp_path / r"testDC.geoh5"

    # Create a workspace
    with Workspace.create(path) as workspace:
        # Create sources along line
        x_loc, y_loc = np.meshgrid(np.arange(n_data), np.arange(-1, 3))
        vertices = np.c_[x_loc.ravel(), y_loc.ravel(), np.zeros_like(x_loc).ravel()]
        parts = np.kron(np.arange(4), np.ones(n_data)).astype("int")
        currents = CurrentElectrode.create(
            workspace, name=name, vertices=vertices, parts=parts
        )
        currents.add_default_ab_cell_id()

        n_dipoles = 9
        dipoles = []
        current_id = []
        for val in currents.ab_cell_id.values:
            cell_id = int(currents.ab_map[val]) - 1

            for dipole in range(n_dipoles):
                dipole_ids = currents.cells[cell_id, :] + 2 + dipole

                if (
                    any(dipole_ids > (vertices.shape[0] - 1))
                    or len(np.unique(parts[np.r_[cell_id, dipole_ids]])) > 1
                ):
                    continue

                dipoles += [dipole_ids]
                current_id += [val]

        potentials = PotentialElectrode.create(
            workspace,
            name=name + "_rx",
            vertices=vertices,
            cells=np.vstack(dipoles).astype("uint32"),
        )

        potentials.add_data(
            {"fake_ab": {"values": np.random.randn(potentials.n_cells)}}
        )
        potentials.ab_cell_id = np.hstack(current_id).astype("int32")
        currents.potential_electrodes = potentials

        # Copy the survey to a new workspace
        path = tmp_path / r"testDC_copy_current.geoh5"
        with Workspace.create(path) as new_workspace:
            new_currents = currents.copy_from_extent(
                np.vstack([[5, 0], [8, 2]]), parent=new_workspace
            )
            new_potentials = potentials.copy_from_extent(
                np.vstack([[7, 0], [11, 2]]), parent=new_workspace
            )

            np.testing.assert_array_almost_equal(
                new_currents.potential_electrodes.vertices, new_potentials.vertices
            )
