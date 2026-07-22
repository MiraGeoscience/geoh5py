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


# mypy: ignore-errors

from __future__ import annotations

import numpy as np
import pytest

from geoh5py.objects import AirborneAppConBaseStations, AirborneAppConReceivers
from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace


def setup_survey(workspace):
    xlocs = np.linspace(-1000, 1000, 10)
    vertices = np.c_[xlocs, np.random.randn(xlocs.shape[0], 2)]
    receivers = AirborneAppConReceivers.create(workspace, vertices=vertices)
    receivers.channels = [30.0, 45.0, 90.0, 180.0, 360.0, 720.0]

    base_stations = AirborneAppConBaseStations.create(workspace, vertices=vertices)
    base_stations.tx_id_property = np.arange(1, base_stations.n_vertices + 1)
    receivers.tx_id_property = np.arange(1, 11)

    receivers.base_stations = base_stations

    return receivers, base_stations


def test_create_survey_airborne_app_con(tmp_path):
    path = tmp_path / f"{__name__}.geoh5"

    workspace = Workspace.create(path)

    receivers, base_stations = setup_survey(workspace)

    assert isinstance(receivers, AirborneAppConReceivers), (
        "Entity type AirborneAppConReceivers failed to create."
    )

    assert isinstance(base_stations, AirborneAppConBaseStations), (
        "Entity type AirborneAppConBaseStations failed to create."
    )

    with pytest.raises(TypeError, match=f"{AirborneAppConBaseStations}"):
        receivers.base_stations = "123"

    with pytest.raises(
        TypeError, match=f"Provided receivers must be of type {type(receivers)}."
    ):
        receivers.receivers = base_stations

    with pytest.raises(TypeError, match=f"{AirborneAppConBaseStations}"):
        base_stations.base_stations = receivers

    with pytest.raises(
        AttributeError,
        match=f"The 'base_station' attribute cannot be set on class {AirborneAppConBaseStations}.",
    ):
        base_stations.base_stations = base_stations

    assert base_stations.base_stations == base_stations

    assert (
        receivers.tx_id_property.entity_type == base_stations.tx_id_property.entity_type
    )
    with pytest.raises(ValueError, match="Mask must be an array of dtype"):
        receivers.copy(mask=np.r_[1, 2, 3])

    assert (
        receivers.copy_from_extent(np.vstack([[1000, 1000], [1001, 1001]])) is None
    ), "Error returning None mask."

    # Re-open the workspace
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


def test_copy(tmp_path):
    # Test copying receiver over through the receivers
    # Create a workspace
    with Workspace.create(tmp_path / f"{__name__}.geoh5") as workspace:
        receivers, base_stations = setup_survey(workspace)

        with Workspace.create(tmp_path / f"{__name__}_copy.geoh5") as new_workspace:
            receivers.copy(parent=new_workspace)

        with Workspace(tmp_path / f"{__name__}_copy.geoh5") as new_workspace:
            receivers_rec = new_workspace.get_entity(
                "Airborne Apparent Conductivity rx"
            )[0]
            compare_entities(
                receivers,
                receivers_rec,
                ignore=["_receivers", "_base_stations", "_parent"],
            )
            compare_entities(
                base_stations,
                receivers_rec.base_stations,
                ignore=["_receivers", "_base_stations", "_parent", "_property_groups"],
            )


def test_copy_by_extent(tmp_path):
    # Test copying receiver over through the base_stations
    # Create a workspace
    with Workspace.create(tmp_path / f"{__name__}.geoh5") as workspace:
        receivers, base_stations = setup_survey(workspace)
        with Workspace.create(tmp_path / f"{__name__}_copy.geoh5") as new_workspace:
            # Test copying receiver over through the base_stations with extent
            base_stations_rec = base_stations.copy_from_extent(
                np.vstack([[0, -np.inf], [2000, np.inf]]), new_workspace
            )

            # Only a sub set of base stations left
            assert base_stations_rec.receivers.n_vertices == np.sum(
                receivers.tx_id_property.values > 5
            )
