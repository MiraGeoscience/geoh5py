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

# pylint: disable=R0914
# pylint: disable=duplicate-code

from __future__ import annotations

import uuid

import numpy as np
import pytest

from geoh5py.objects import CurrentElectrode, PotentialElectrode
from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace


def test_create_survey_dcip(tmp_path):
    name = "TestCurrents"
    n_data = 12
    path = tmp_path / r"testDC.geoh5"

    # Create a workspace
    workspace = Workspace.create_geoh5(path)

    # Create sources along line
    x_loc, y_loc = np.meshgrid(np.arange(n_data), np.arange(-1, 3))
    vertices = np.c_[x_loc.ravel(), y_loc.ravel(), np.zeros_like(x_loc).ravel()]
    parts = np.kron(np.arange(4), np.ones(n_data)).astype("int")
    currents = CurrentElectrode.create(
        workspace, name=name, vertices=vertices, parts=parts
    )
    currents.add_default_ab_cell_id()
    potentials = PotentialElectrode.create(
        workspace, name=name + "_rx", vertices=vertices
    )
    n_dipoles = 9
    dipoles = []
    current_id = []
    for val in currents.ab_cell_id.values:
        cell_id = int(currents.ab_map[val]) - 1

        for dipole in range(n_dipoles):
            dipole_ids = currents.cells[cell_id, :] + 2 + dipole

            if (
                any(dipole_ids > (potentials.n_vertices - 1))
                or len(np.unique(parts[dipole_ids])) > 1
            ):
                continue

            dipoles += [dipole_ids]
            current_id += [val]

    potentials.cells = np.vstack(dipoles).astype("uint32")

    fake_ab = potentials.add_data(
        {"fake_ab": {"values": np.random.randn(potentials.n_cells)}}
    )

    with pytest.raises(TypeError):
        potentials.ab_cell_id = fake_ab

    potentials.ab_cell_id = np.hstack(current_id).astype("int32")

    # Change again only the values
    ab_data = potentials.get_data("A-B Cell ID")[0]
    new_values = ab_data.values
    new_values[0] = 5
    potentials.ab_cell_id = new_values

    assert (
        len(potentials.get_data("A-B Cell ID")) == 1
    ), "Issue with new A-B Cell ID data created"

    fake_meta = {
        "Current Electrodes": uuid.uuid4(),
        "Potential Electrodes": uuid.uuid4(),
        "One too many key": uuid.uuid4(),
    }
    with pytest.raises(ValueError):
        potentials.metadata = fake_meta

    del fake_meta["One too many key"]

    with pytest.raises(IndexError):
        potentials.metadata = fake_meta

    fake_meta["Current Electrodes"] = currents.uid

    with pytest.raises(IndexError):
        potentials.metadata = fake_meta

    fake_meta["Potential Electrodes"] = potentials.uid

    potentials.current_electrodes = currents
    assert (
        currents.potential_electrodes == potentials
    ), "Error assigning the potentiel_electrodes."
    assert (
        potentials.current_electrodes == currents
    ), "Error assigning the current_electrodes."

    assert (
        currents.metadata
        == potentials.metadata
        == {
            "Current Electrodes": currents.uid,
            "Potential Electrodes": potentials.uid,
        }
    ), "Error assigning metadata"

    # Repeat the other way
    with pytest.raises(TypeError) as info:
        potentials.current_electrodes = None
    assert info.type == TypeError, "Code did not catch TypeError"

    with pytest.raises(TypeError) as info:
        currents.potential_electrodes = None
    assert info.type == TypeError, "Code did not catch TypeError"

    setattr(potentials, "_current_electrodes", None)
    setattr(currents, "_potential_electrodes", None)

    currents.potential_electrodes = potentials
    assert (
        currents.potential_electrodes == potentials
    ), "Error assigning the potentiel_electrodes."
    assert (
        potentials.current_electrodes == currents
    ), "Error assigning the current_electrodes."

    assert (
        currents.metadata
        == potentials.metadata
        == {
            "Current Electrodes": currents.uid,
            "Potential Electrodes": potentials.uid,
        }
    ), "Error assigning metadata"

    # Re-open the workspace and read data back in
    new_workspace = Workspace(path)
    currents_rec = new_workspace.get_entity(name)[0]
    potentials_rec = new_workspace.get_entity(name + "_rx")[0]

    # Check entities
    compare_entities(
        currents, currents_rec, ignore=["_potential_electrodes", "_parent"]
    )
    compare_entities(
        potentials, potentials_rec, ignore=["_current_electrodes", "_parent"]
    )
