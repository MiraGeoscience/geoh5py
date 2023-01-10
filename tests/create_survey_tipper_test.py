#  Copyright (c) 2023 Mira Geoscience Ltd Ltd.
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

# mypy: ignore-errors

from __future__ import annotations

import numpy as np
import pytest

from geoh5py.objects import TipperBaseStations, TipperReceivers
from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace


def test_create_survey_tipper(tmp_path):

    path = tmp_path / r"../testTipper.geoh5"

    workspace = Workspace(path)
    xlocs = np.linspace(-1000, 1000, 10)
    vertices = np.c_[xlocs, np.random.randn(xlocs.shape[0], 2)]
    receivers = TipperReceivers.create(workspace, vertices=vertices)
    receivers.channels = [30.0, 45.0, 90.0, 180.0, 360.0, 720.0]
    assert isinstance(
        receivers, TipperReceivers
    ), "Entity type TipperReceivers failed to create."
    base_stations = TipperBaseStations.create(workspace, vertices=vertices)
    assert isinstance(
        base_stations, TipperBaseStations
    ), "Entity type TipperBaseStations failed to create."

    with pytest.warns(UserWarning) as warn:
        print(receivers.base_stations)

    assert "Associated `base_stations` entity not set." in str(warn[0])

    with pytest.raises(TypeError) as error:
        receivers.base_stations = "123"

    assert f"{TipperBaseStations}" in str(
        error
    ), "Missed raising error on 'base stations' change."

    with pytest.raises(
        TypeError, match=f"Provided receivers must be of type {type(receivers)}."
    ):
        receivers.receivers = base_stations

    with pytest.raises(TypeError, match=f"{TipperBaseStations}"):
        base_stations.base_stations = receivers

    with pytest.raises(AttributeError) as error:
        base_stations.base_stations = base_stations

    assert (
        f"The 'base_station' attribute cannot be set on class {TipperBaseStations}."
        in str(error)
    ), "Missed raising AttributeError on setting 'base_stations' on self."

    assert base_stations.base_stations == base_stations

    base_stations_test = TipperBaseStations.create(workspace, vertices=vertices[1:, :])

    with pytest.raises(AttributeError) as error:
        receivers.base_stations = base_stations_test

    assert "The input 'base_stations' should have n_vertices" in str(error)

    receivers.base_stations = base_stations

    new_workspace = Workspace(path)
    base_stations_rec = new_workspace.get_entity(base_stations.uid)[0]
    receivers_rec = new_workspace.get_entity(receivers.uid)[0]

    assert receivers_rec.default_input_types == ["Rx and base stations"]

    # Check entities
    compare_entities(
        base_stations,
        base_stations_rec,
        ignore=["_receivers", "_base_stations", "_parent"],
    )
    compare_entities(
        receivers,
        receivers_rec,
        ignore=["_receivers", "_base_stations", "_parent", "_property_groups"],
    )

    # Test copying receiver over through the receivers
    # Create a workspace
    receivers.copy(Workspace(tmp_path / r"testATEM_copy.geoh5"))
    new_workspace = Workspace(tmp_path / r"testATEM_copy.geoh5")
    receivers_rec = new_workspace.get_entity("TipperReceivers")[0]
    compare_entities(
        receivers, receivers_rec, ignore=["_receivers", "_base_stations", "_parent"]
    )
    compare_entities(
        base_stations,
        receivers_rec.base_stations,
        ignore=["_receivers", "_base_stations", "_parent", "_property_groups"],
    )

    # Test copying receiver over through the base_stations
    # Create a workspace
    new_workspace = Workspace(tmp_path / r"testATEM_copy2.geoh5")
    base_stations_rec = base_stations.copy(new_workspace)
    compare_entities(
        receivers,
        base_stations_rec.receivers,
        ignore=["_receivers", "_base_stations", "_parent"],
    )
    compare_entities(
        base_stations,
        base_stations_rec,
        ignore=["_receivers", "_base_stations", "_parent", "_property_groups"],
    )
