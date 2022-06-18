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

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from geoh5py.objects import AirborneTEMReceivers, AirborneTEMTransmitters
from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace


def test_create_survey_tem(tmp_path):

    name = "Survey"
    path = Path(tmp_path) / r"../testATEM.geoh5"

    # Create a workspace
    workspace = Workspace(path)
    xlocs = np.linspace(-1000, 1000, 10)
    vertices = np.c_[xlocs, np.random.randn(xlocs.shape[0], 2)]
    receivers = AirborneTEMReceivers.create(
        workspace, vertices=vertices, name=name + "_rx"
    )
    assert isinstance(
        receivers, AirborneTEMReceivers
    ), "Entity type AirborneTEMReceivers failed to create."
    transmitters = AirborneTEMTransmitters.create(
        workspace, vertices=vertices, name=name + "_tx"
    )
    assert isinstance(
        transmitters, AirborneTEMTransmitters
    ), "Entity type AirborneTEMTransmitters failed to create."

    with pytest.raises(TypeError) as error:
        receivers.transmitters = "123"

    assert f" must be of type {AirborneTEMTransmitters}" in str(
        error
    ), "Missed raising error on 'transmitter' change."

    with pytest.raises(AttributeError) as error:
        receivers.receivers = transmitters

    assert f"The 'receivers' attribute cannot be set on class {type(receivers)}" in str(
        error
    ), "Missed raising AttributeError on setting 'receivers' on self."

    with pytest.raises(AttributeError) as error:
        transmitters.transmitters = receivers

    assert (
        f"The 'transmitters' attribute cannot be set on class {type(transmitters)}"
        in str(error)
    ), "Missed raising AttributeError on setting 'transmitters' on self."

    receivers.transmitters = transmitters

    with pytest.raises(TypeError) as error:
        receivers.loop_radius = "123"

    assert "Input 'loop_radius' must be of type 'float'" in str(
        error
    ), "Failed TypeError on loop_radius."

    receivers.loop_radius = 123.0
    angles = receivers.add_data(
        {"angles": {"values": np.random.randn(receivers.n_vertices)}}
    )
    for key in [
        "pitch",
        "roll",
        "yaw",
        "inline_offset",
        "crossline_offset",
        "vertical_offset",
    ]:
        with pytest.raises(TypeError) as error:
            setattr(receivers, key, "abc")

        assert f"Input '{key}' must be one of type float, uuid.UUID or None" in str(
            error
        ), f"Missed raising error on type of '{key}'."

        setattr(receivers, key, angles.uid)
        assert (
            f"{key.capitalize().replace('_', ' ')} property"
            in receivers.metadata["EM Dataset"]
        ), f"Wrong metadata label set on '{key}' for input uuid."

        assert (
            getattr(receivers, key) == angles.uid
        ), f"Wrong metadata assignment on {key} property."
        assert (
            f"{key.capitalize().replace('_', ' ')} value"
            not in receivers.metadata["EM Dataset"]
        ), f"Failed in removing '{key}' value from metadata."

        setattr(receivers, key, None)
        assert getattr(receivers, key) is None
        setattr(receivers, key, 3.0)
        assert (
            f"{key.capitalize().replace('_', ' ')} value"
            in receivers.metadata["EM Dataset"]
        ), f"Wrong metadata label set on '{key}' for input uuid."
        assert (
            getattr(receivers, key) == 3.0
        ), f"Wrong metadata assignment on {key} value."
        assert (
            f"{key.capitalize().replace('_', ' ')} property"
            not in receivers.metadata["EM Dataset"]
        ), f"Failed in removing '{key}' property from metadata."

    assert (
        getattr(receivers, "relative_to_bearing", None) is None
    ), "Default 'relative_to_bearing' should be None."

    with pytest.raises(TypeError) as error:
        receivers.relative_to_bearing = "nan"

    assert "Input 'relative_to_bearing' must be one of type 'bool'" in str(
        error
    ), "Failed TypeError."

    receivers.relative_to_bearing = True

    assert getattr(
        receivers, "relative_to_bearing", None
    ), "Failed setting 'relative_to_bearing' to True."

    new_workspace = Workspace(path)
    transmitters_rec = new_workspace.get_entity(name + "_tx")[0]
    receivers_rec = new_workspace.get_entity(name + "_rx")[0]

    # Check entities
    compare_entities(
        transmitters,
        transmitters_rec,
        ignore=["_receivers", "_transmitters", "_parent"],
    )
    compare_entities(
        receivers,
        receivers_rec,
        ignore=["_receivers", "_transmitters", "_parent", "_property_groups"],
    )

    # Test copying receiver over through the receivers
    # Create a workspace
    new_workspace = Workspace(Path(tmp_path) / r"testATEM_copy.geoh5")
    receivers_rec = receivers.copy(new_workspace)
    compare_entities(
        receivers, receivers_rec, ignore=["_receivers", "_transmitters", "_parent"]
    )
    compare_entities(
        transmitters,
        receivers_rec.transmitters,
        ignore=["_receivers", "_transmitters", "_parent", "_property_groups"],
    )

    # Test copying receiver over through the transmitters
    # Create a workspace
    new_workspace = Workspace(Path(tmp_path) / r"testATEM_copy2.geoh5")
    transmitters_rec = transmitters.copy(new_workspace)
    compare_entities(
        receivers,
        transmitters_rec.receivers,
        ignore=["_receivers", "_transmitters", "_parent"],
    )
    compare_entities(
        transmitters,
        transmitters_rec,
        ignore=["_receivers", "_transmitters", "_parent", "_property_groups"],
    )


def test_survey_tem_data(tmp_path):

    name = "Survey"
    path = Path(tmp_path) / r"../testATEM.geoh5"

    # Create a workspace
    workspace = Workspace(path)
    receivers = workspace.get_entity(name + "_rx")[0]
    transmitters = receivers.transmitters

    # Add channels
    with pytest.raises(TypeError) as error:
        receivers.channels = {"abc": 1, "dfg": 2}

    assert f"Values provided as 'channels' must be a list of {float}" in str(
        error
    ), "Missed raising error on 'channels' change."

    channels = np.logspace(-3, -2, 10)
    receivers.channels = channels

    assert np.all(
        receivers.channels == receivers.metadata["EM Dataset"]["Channels"] == channels
    ), "Channels values did not get set properly."

    # Add data as a list of FloatData
    data = receivers.add_data(
        {
            f"Channel_{ii}": {"values": np.random.randn(receivers.n_vertices)}
            for ii in receivers.channels
        }
    )

    with pytest.raises(ValueError) as error:
        receivers.add_components_data({"time_data": data[1:]})

    assert "The number of channel values provided" in str(
        error
    ), "Failed to check length of input"

    prop_group = receivers.add_components_data({"time_data": data})[0]

    assert (
        prop_group.name in receivers.metadata["EM Dataset"]["Property groups"]
    ), "Failed to add the property group to metadata from 'add_components_data' method."

    assert receivers.components == {
        "time_data": data
    }, "Property 'components' not accessing metadata."

    with pytest.raises(ValueError) as error:
        receivers.add_components_data({"time_data": data})

    assert (
        "PropertyGroup named 'time_data' already exists on the survey entity."
    ) in str(
        error
    ), "Failed to protect against creation of PropertyGroup with same name."

    # Create another property group and assign by name
    prop_group = receivers.add_data_to_group(data, "NewGroup")

    receivers.edit_metadata({"Property groups": prop_group})

    assert (
        prop_group.name in receivers.metadata["EM Dataset"]["Property groups"]
        and len(receivers.metadata["EM Dataset"]["Property groups"]) == 2
    ), "Failed to add the property group to list of metadata."

    receivers.edit_metadata({"Property groups": None})

    assert (
        len(receivers.metadata["EM Dataset"]["Property groups"]) == 0
    ), "Failed to remove property groups from the metadata."

    with pytest.raises(TypeError) as error:
        receivers.edit_metadata({"Property groups": 1234})

    assert "Input value for 'Property groups' must be a PropertyGroup" in str(
        error
    ), "Failed to detect property group type error."

    with pytest.raises(TypeError) as error:
        receivers.add_components_data(
            {"new_times": [["abc"]] * len(receivers.channels)}
        )

    assert (
        "List of values provided for component 'new_times' must be a list of "
    ) in str(error), "Failed to protect against TypeError on add_components_data"

    with pytest.raises(ValueError) as error:
        receivers.unit = "hello world"

    assert "Input 'unit' must be one of" in str(
        error
    ), "Missed raising error on 'unit' change."

    receivers.unit = "Seconds (s)"

    with pytest.raises(ValueError) as error:
        receivers.waveform = np.ones(3)

    assert "Input waveform must be a numpy.ndarray of shape (*, 2)." in str(
        error
    ), "Missed raising error shape of 'waveform'."

    with pytest.raises(TypeError) as error:
        receivers.waveform = [1, 2, 3]

    assert "Input waveform must be a numpy.ndarray or None." in str(
        error
    ), "Missed raising error on type of 'waveform'."

    waveform = np.c_[np.logspace(-5, -1.9, 10), np.linspace(0, 1, 10)]
    receivers.waveform = waveform

    with pytest.raises(ValueError) as error:
        receivers.timing_mark = "abc"

    assert "Input timing_mark must be a float or None." in str(
        error
    ), "Missed raising error on type of 'timing_mark'."
    receivers.timing_mark = 10**-3.1

    assert (
        getattr(receivers, "timing_mark") == 10**-3.1
    ), "Failed in setting 'timing_mark'."
    assert (
        receivers.metadata == transmitters.metadata
    ), "Error synchronizing the transmitters and receivers metadata."

    receivers.timing_mark = None

    assert (
        "Timing mark" not in receivers.metadata["EM Dataset"]["Waveform"]
    ), "Error removing the timing mark."

    # Repeat with timing mark first.
    receivers.waveform = None
    assert "Waveform" not in receivers.metadata["EM Dataset"]
    receivers.timing_mark = 10**-3.1
    receivers.waveform = waveform

    new_workspace = Workspace(path)

    receivers_rec = new_workspace.get_entity(name + "_rx")[0]
    np.testing.assert_almost_equal(receivers_rec.waveform, waveform)

    new_workspace = Workspace(Path(tmp_path) / r"testATEM_copy2.geoh5")
    receivers_rec = receivers.copy(new_workspace)
    compare_entities(
        receivers, receivers_rec, ignore=["_receivers", "_transmitters", "_parent"]
    )
