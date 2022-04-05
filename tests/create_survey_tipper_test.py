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

import os

import numpy as np
import pytest

from geoh5py.objects import TipperBaseStations, TipperReceivers
from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace


def test_create_survey_tipper(tmp_path):
    # workspace = Workspace("ztem.geoh5")
    # survey = workspace.get_entity(
    #     "Inv_North_ref2em2_bound1e7_scottMeshTopo_everyIter_60pct rx"
    # )[0]
    # print(tmp_path, survey)
    # name = "Survey"
    path = os.path.join(tmp_path, r"../testTipper.geoh5")
    #
    # # Create a workspace
    workspace = Workspace(path)
    xlocs = np.linspace(-1000, 1000, 10)
    vertices = np.c_[xlocs, np.random.randn(xlocs.shape[0], 2)]
    receivers = TipperReceivers.create(workspace, vertices=vertices)
    assert isinstance(
        receivers, TipperReceivers
    ), "Entity type TipperReceivers failed to create."
    base_stations = TipperBaseStations.create(workspace, vertices=vertices)
    assert isinstance(
        base_stations, TipperBaseStations
    ), "Entity type TipperBaseStations failed to create."

    with pytest.raises(TypeError) as error:
        receivers.base_stations = "123"

    assert f"{TipperBaseStations}" in str(
        error
    ), "Missed raising error on 'base stations' change."

    with pytest.raises(AttributeError) as error:
        receivers.receivers = base_stations

    assert (
        "Attribute 'receivers' of the class 'TipperReceivers' must reference to self."
        in str(error)
    ), "Missed raising AttributeError on setting 'receivers' on self."

    with pytest.raises(AttributeError) as error:
        base_stations.base_stations = receivers

    assert (
        "Attribute 'base_stations' of the class 'TipperBaseStations' must reference to self."
        in str(error)
    ), "Missed raising AttributeError on setting 'base_stations' on self."

    receivers.base_stations = base_stations

    workspace.finalize()

    new_workspace = Workspace(path)
    base_stations_rec = new_workspace.get_entity("TipperBaseStations")[0]
    receivers_rec = new_workspace.get_entity("TipperReceivers")[0]

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
    #
    # # Test copying receiver over through the receivers
    # # Create a workspace
    # new_workspace = Workspace(Path(tmp_path) / r"testATEM_copy.geoh5")
    # receivers_rec = receivers.copy(new_workspace)
    # compare_entities(
    #     receivers, receivers_rec, ignore=["_receivers", "_base_stations", "_parent"]
    # )
    # compare_entities(
    #     base_stations,
    #     receivers_rec.base_stations,
    #     ignore=["_receivers", "_base_stations", "_parent", "_property_groups"],
    # )
    #
    # # Test copying receiver over through the base_stations
    # # Create a workspace
    # new_workspace = Workspace(Path(tmp_path) / r"testATEM_copy2.geoh5")
    # base_stations_rec = base_stations.copy(new_workspace)
    # compare_entities(
    #     receivers,
    #     base_stations_rec.receivers,
    #     ignore=["_receivers", "_base_stations", "_parent"],
    # )
    # compare_entities(
    #     base_stations,
    #     base_stations_rec,
    #     ignore=["_receivers", "_base_stations", "_parent", "_property_groups"],
    # )


def test_survey_tipper_data(tmp_path):
    print(tmp_path)
    # name = "Survey"
    # path = Path(tmp_path) / r"../testATEM.geoh5"
    #
    # # Create a workspace
    # workspace = Workspace(path)
    # receivers = workspace.get_entity(name + "_rx")[0]
    # base_stations = receivers.base_stations
    #
    # # Add channels
    # with pytest.raises(TypeError) as error:
    #     receivers.channels = {"abc": 1, "dfg": 2}
    #
    # assert f"Values provided as 'channels' must be a list of {float}" in str(
    #     error
    # ), "Missed raising error on 'channels' change."
    #
    # channels = np.logspace(-3, -2, 10)
    # receivers.channels = channels
    #
    # assert np.all(
    #     receivers.channels == receivers.metadata["EM Dataset"]["Channels"] == channels
    # ), "Channels values did not get set properly."
    #
    # # Add data as a list of FloatData
    # data = receivers.add_data(
    #     {
    #         f"Channel_{ii}": {"values": np.random.randn(receivers.n_vertices)}
    #         for ii in receivers.channels
    #     }
    # )
    #
    # with pytest.raises(ValueError) as error:
    #     receivers.add_components_data({"time_data": data[1:]})
    #
    # assert "The number of channel values provided" in str(
    #     error
    # ), "Failed to check length of input"
    #
    # prop_group = receivers.add_components_data({"time_data": data})[0]
    #
    # assert (
    #     prop_group.name in receivers.metadata["EM Dataset"]["Property groups"]
    # ), "Failed to add the property group to metadata from 'add_components_data' method."
    #
    # assert receivers.components == {
    #     "time_data": data
    # }, "Property 'components' not accessing metadata."
    #
    # with pytest.raises(ValueError) as error:
    #     receivers.add_components_data({"time_data": data})
    #
    # assert (
    #     "PropertyGroup named 'time_data' already exists on the survey entity."
    # ) in str(
    #     error
    # ), "Failed to protect against creation of PropertyGroup with same name."
    #
    # # Create another property group and assign by name
    # prop_group = receivers.add_data_to_group(data, "NewGroup")
    #
    # receivers.edit_metadata({"Property groups": prop_group})
    #
    # assert (
    #     prop_group.name in receivers.metadata["EM Dataset"]["Property groups"]
    #     and len(receivers.metadata["EM Dataset"]["Property groups"]) == 2
    # ), "Failed to add the property group to list of metadata."
    #
    # receivers.edit_metadata({"Property groups": None})
    #
    # assert (
    #     len(receivers.metadata["EM Dataset"]["Property groups"]) == 0
    # ), "Failed to remove property groups from the metadata."
    #
    # with pytest.raises(TypeError) as error:
    #     receivers.edit_metadata({"Property groups": 1234})
    #
    # assert "Input value for 'Property groups' must be a PropertyGroup" in str(
    #     error
    # ), "Failed to detect property group type error."
    #
    # with pytest.raises(TypeError) as error:
    #     receivers.add_components_data(
    #         {"new_times": [["abc"]] * len(receivers.channels)}
    #     )
    #
    # assert (
    #     "List of values provided for component 'new_times' must be a list of "
    # ) in str(error), "Failed to protect against TypeError on add_components_data"
    #
    # with pytest.raises(ValueError) as error:
    #     receivers.unit = "hello world"
    #
    # assert "Input 'unit' must be one of" in str(
    #     error
    # ), "Missed raising error on 'unit' change."
    #
    # receivers.unit = "Seconds (s)"
    #
    # with pytest.raises(ValueError) as error:
    #     receivers.waveform = np.ones(3)
    #
    # assert "Input waveform must be a numpy.ndarray of shape (*, 2)." in str(
    #     error
    # ), "Missed raising error shape of 'waveform'."
    #
    # with pytest.raises(TypeError) as error:
    #     receivers.waveform = [1, 2, 3]
    #
    # assert "Input waveform must be a numpy.ndarray or None." in str(
    #     error
    # ), "Missed raising error on type of 'waveform'."
    #
    # waveform = np.c_[np.logspace(-5, -1.9, 10), np.linspace(0, 1, 10)]
    # receivers.waveform = waveform
    #
    # with pytest.raises(ValueError) as error:
    #     receivers.timing_mark = "abc"
    #
    # assert "Input timing_mark must be a float or None." in str(
    #     error
    # ), "Missed raising error on type of 'timing_mark'."
    # receivers.timing_mark = 10**-3.1
    #
    # assert (
    #     getattr(receivers, "timing_mark") == 10**-3.1
    # ), "Failed in setting 'timing_mark'."
    # assert (
    #     receivers.metadata == base_stations.metadata
    # ), "Error synchronizing the base_stations and receivers metadata."
    #
    # receivers.timing_mark = None
    #
    # assert (
    #     "Timing mark" not in receivers.metadata["EM Dataset"]["Waveform"]
    # ), "Error removing the timing mark."
    #
    # workspace.finalize()
    # new_workspace = Workspace(path)
    #
    # receivers_rec = new_workspace.get_entity(name + "_rx")[0]
    # np.testing.assert_almost_equal(receivers_rec.waveform, waveform)
    #
    # new_workspace = Workspace(Path(tmp_path) / r"testATEM_copy2.geoh5")
    # receivers_rec = receivers.copy(new_workspace)
    # compare_entities(
    #     receivers, receivers_rec, ignore=["_receivers", "_base_stations", "_parent"]
    # )
