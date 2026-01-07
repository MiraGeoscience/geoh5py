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

from geoh5py.groups import PropertyGroup
from geoh5py.objects import (
    AirborneFEMReceivers,
    AirborneFEMTransmitters,
    LargeLoopGroundFEMReceivers,
    LargeLoopGroundFEMTransmitters,
    MovingLoopGroundFEMReceivers,
    MovingLoopGroundFEMTransmitters,
)
from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace


def test_create_survey_airborne_fem(tmp_path):
    name = "Survey"
    path = Path(tmp_path) / r"../testAFEM.geoh5"

    # Create a workspace
    workspace = Workspace.create(path, mode="r+")
    xlocs = np.linspace(-1000, 1000, 10)
    vertices = np.c_[xlocs, np.random.randn(xlocs.shape[0], 2)]
    receivers = AirborneFEMReceivers.create(
        workspace, vertices=vertices, name=name + "_rx"
    )
    receivers.tx_id_property = np.arange(receivers.n_vertices)

    assert isinstance(receivers, AirborneFEMReceivers), (
        "Entity type AirborneFEMReceivers failed to create."
    )
    transmitters = AirborneFEMTransmitters.create(
        workspace, vertices=vertices + 10.0, name=name + "_tx"
    )
    transmitters.tx_id_property = np.arange(transmitters.n_vertices)

    assert isinstance(transmitters, AirborneFEMTransmitters), (
        "Entity type AirborneFEMTransmitters failed to create."
    )

    with pytest.raises(TypeError, match=f" must be of type {AirborneFEMTransmitters}"):
        receivers.transmitters = "123"

    with pytest.raises(
        TypeError,
        match=f"Provided receivers must be of type {type(receivers)}",
    ):
        receivers.receivers = transmitters

    with pytest.raises(
        TypeError,
        match=f"Provided transmitters must be of type {type(transmitters)}",
    ):
        transmitters.transmitters = receivers

    transmitters.receivers = receivers

    assert (
        transmitters.tx_id_property.entity_type == receivers.tx_id_property.entity_type
    )

    with pytest.raises(TypeError, match="Input 'loop_radius' must be of type 'float'"):
        receivers.loop_radius = "123"

    # Reset the tx_id_property
    receivers.tx_id_property = np.arange(receivers.n_vertices)
    transmitters.tx_id_property = np.arange(transmitters.n_vertices)

    assert (
        receivers.metadata["EM Dataset"]["Tx ID tx property"]
        == transmitters.tx_id_property.uid
    )
    assert (
        receivers.tx_id_property.entity_type == transmitters.tx_id_property.entity_type
    )

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
        with pytest.raises(
            TypeError,
            match=f"Input '{key}' must be one of type float, uuid.UUID or None",
        ):
            setattr(receivers, key, "abc")

        setattr(receivers, key, angles.uid)
        assert (
            f"{key.capitalize().replace('_', ' ')} property"
            in receivers.metadata["EM Dataset"]
        ), f"Wrong metadata label set on '{key}' for input uuid."

        assert getattr(receivers, key) == angles.uid, (
            f"Wrong metadata assignment on {key} property."
        )
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
        assert getattr(receivers, key) == 3.0, (
            f"Wrong metadata assignment on {key} value."
        )
        assert (
            f"{key.capitalize().replace('_', ' ')} property"
            not in receivers.metadata["EM Dataset"]
        ), f"Failed in removing '{key}' property from metadata."

    assert getattr(receivers, "relative_to_bearing", None) is None, (
        "Default 'relative_to_bearing' should be None."
    )

    with pytest.raises(
        TypeError, match="Input 'relative_to_bearing' must be one of type 'bool'"
    ):
        receivers.relative_to_bearing = "nan"

    receivers.relative_to_bearing = True

    assert getattr(receivers, "relative_to_bearing", None), (
        "Failed setting 'relative_to_bearing' to True."
    )

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
    new_workspace = Workspace.create(Path(tmp_path) / r"testAFEM_copy.geoh5")
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
    new_workspace = Workspace.create(Path(tmp_path) / r"testAFEM_copy2.geoh5")
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


def test_survey_airborne_fem_data(tmp_path):
    name = "Survey"
    path = Path(tmp_path) / r"../testAFEM.geoh5"

    # Create a workspace
    workspace = Workspace(path)
    receivers = workspace.get_entity(name + "_rx")[0]
    transmitters = receivers.transmitters

    # Add channels
    with pytest.raises(
        TypeError, match=f"Values provided as 'channels' must be a list of {float}"
    ):
        receivers.channels = {"abc": 1, "dfg": 2}

    channels = np.array([0.01, 1.0, 100.0])
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

    with pytest.raises(ValueError, match="The number of channel values provided"):
        receivers.add_components_data({"time_data": data[1:]})

    prop_group = receivers.add_components_data({"time_data": data})[0]

    assert prop_group.name in receivers.metadata["EM Dataset"]["Property groups"], (
        "Failed to add the property group to metadata from 'add_components_data' method."
    )

    assert receivers.components == {"time_data": data}, (
        "Property 'components' not accessing metadata."
    )

    with pytest.raises(
        ValueError,
        match="PropertyGroup named 'time_data' already exists on the survey entity.",
    ):
        receivers.add_components_data({"time_data": data})

    # Create another property group and assign by name
    prop_group = receivers.add_data_to_group(data, "NewGroup")

    receivers.edit_em_metadata({"Property groups": prop_group})

    assert (
        prop_group.name in receivers.metadata["EM Dataset"]["Property groups"]
        and len(receivers.metadata["EM Dataset"]["Property groups"]) == 2
    ), "Failed to add the property group to list of metadata."

    receivers.edit_em_metadata({"Property groups": None})

    assert len(receivers.metadata["EM Dataset"]["Property groups"]) == 0, (
        "Failed to remove property groups from the metadata."
    )

    with pytest.raises(
        TypeError, match="Input value for 'Property groups' must be a PropertyGroup"
    ):
        receivers.edit_em_metadata({"Property groups": 1234})

    with pytest.raises(
        TypeError,
        match="List of values provided for component 'new_times' must be a list of ",
    ):
        receivers.add_components_data(
            {"new_times": [["abc"]] * len(receivers.channels)}
        )

    with pytest.raises(ValueError, match="Input 'unit' must be one of"):
        receivers.unit = "hello world"

    receivers.unit = "Hertz (Hz)"

    assert receivers.metadata == transmitters.metadata, (
        "Error synchronizing the transmitters and receivers metadata."
    )

    with pytest.raises(ValueError, match="Mask must be an array of dtype"):
        receivers.copy(mask=np.r_[1, 2, 3])

    workspace.close()

    # Test copying receiver over through the receivers
    with Workspace(path) as workspace:
        receivers_orig = workspace.get_entity(name + "_rx")[0]

        with Workspace.create(
            Path(tmp_path) / r"testAFEM_copy2.geoh5"
        ) as new_workspace:
            receivers_rec = receivers_orig.copy(new_workspace)
            compare_entities(
                receivers_orig,
                receivers_rec,
                ignore=["_receivers", "_transmitters", "_parent", "_property_groups"],
            )

        with Workspace.create(
            Path(tmp_path) / r"testAFEM_copy_extent.geoh5"
        ) as new_workspace:
            receivers_rec = receivers_orig.copy_from_extent(
                np.vstack([[0, -5], [1500, 5]]), parent=new_workspace
            )
            assert receivers_rec.n_vertices == receivers_rec.transmitters.n_vertices
            np.testing.assert_almost_equal(
                receivers_orig.vertices[5:, :], receivers_rec.vertices
            )

            for child_a, child_b in zip(
                [
                    child
                    for child in receivers_orig.children
                    if not isinstance(child, PropertyGroup)
                ],
                [
                    child
                    for child in receivers_rec.children
                    if not isinstance(child, PropertyGroup)
                ],
                strict=False,
            ):
                np.testing.assert_almost_equal(child_a.values[5:], child_b.values)


def test_create_survey_ground_fem_large_loop(
    tmp_path,
):  # pylint: disable=too-many-locals
    path = Path(tmp_path) / r"groundFEM.geoh5"

    # Create a workspace
    workspace = Workspace.create(path)

    vertices = []
    tx_loops = []
    tx_id = []
    tx_cells = []
    count = 0
    for ind in range(2):
        offset = 500.0 * ind
        xlocs = np.linspace(-1000, 1000, 10)

        vertices += [np.c_[xlocs, np.zeros_like(xlocs) + offset, np.zeros_like(xlocs)]]
        tx_id += [np.ones_like(xlocs) * (ind + 1)]
        tx_locs = np.r_[
            np.c_[-100, -100],
            np.c_[-100, 100],
            np.c_[100, 100],
            np.c_[100, -100],
        ]
        tx_loops += [np.c_[tx_locs[:, 0], tx_locs[:, 1] + offset, np.zeros(4)]]
        tx_cells += [np.c_[np.arange(3) + count, np.arange(3) + count + 1]]
        tx_cells += [np.c_[count + 3, count]]
        count += 4

    receivers = LargeLoopGroundFEMReceivers.create(
        workspace, vertices=np.vstack(vertices)
    )
    assert isinstance(receivers, LargeLoopGroundFEMReceivers), (
        "Entity type GroundFEMReceiversLargeLoop failed to create."
    )

    transmitters = LargeLoopGroundFEMTransmitters.create(
        workspace,
        vertices=np.vstack(tx_loops),
        cells=np.vstack(tx_cells),
    )
    transmitters.tx_id_property = transmitters.parts + 1

    assert isinstance(transmitters, LargeLoopGroundFEMTransmitters), (
        "Entity type GroundFEMTransmittersLargeLoop failed to create."
    )

    with pytest.raises(
        TypeError, match=f" must be of type {LargeLoopGroundFEMTransmitters}"
    ):
        receivers.transmitters = "123"

    with pytest.raises(
        TypeError,
        match=f"Provided receivers must be of type {type(receivers)}",
    ):
        receivers.receivers = transmitters

    with pytest.raises(
        TypeError,
        match=f"Provided transmitters must be of type {type(transmitters)}",
    ):
        transmitters.transmitters = receivers
    receivers.tx_id_property = np.hstack(tx_id)
    receivers.transmitters = transmitters

    assert (
        receivers.tx_id_property.entity_type == transmitters.tx_id_property.entity_type
    )

    with Workspace.create(Path(tmp_path) / r"testGround_copy.geoh5") as new_workspace:
        receivers_orig = receivers.copy(new_workspace)
        transmitters_rec = receivers.transmitters.copy_from_extent(
            np.vstack([[-150, -150], [150, 150]]), parent=new_workspace
        )
        assert transmitters_rec.receivers.n_vertices == receivers_orig.n_vertices / 2.0
        assert (
            transmitters_rec.n_vertices == receivers_orig.transmitters.n_vertices / 2.0
        )


def test_create_survey_ground_fem(tmp_path):
    name = "Survey"
    path = Path(tmp_path) / r"../testGFEM.geoh5"

    # Create a workspace
    workspace = Workspace.create(path)
    xlocs = np.linspace(-1000, 1000, 10)
    vertices = np.c_[xlocs, np.random.randn(xlocs.shape[0], 2)]
    receivers = MovingLoopGroundFEMReceivers.create(
        workspace, vertices=vertices, name=name + "_rx"
    )
    assert isinstance(receivers, MovingLoopGroundFEMReceivers), (
        "Entity type MovingLoopGroundFEMReceivers failed to create."
    )
    transmitters = MovingLoopGroundFEMTransmitters.create(
        workspace, vertices=vertices + 10.0, name=name + "_tx"
    )
    assert isinstance(transmitters, MovingLoopGroundFEMTransmitters), (
        "Entity type MovingLoopGroundFEMTransmitters failed to create."
    )

    with pytest.raises(
        TypeError, match=f" must be of type {MovingLoopGroundFEMTransmitters}"
    ):
        receivers.transmitters = "123"

    with pytest.raises(
        TypeError,
        match=f"Provided receivers must be of type {type(receivers)}",
    ):
        receivers.receivers = transmitters

    with pytest.raises(
        TypeError,
        match=f"Provided transmitters must be of type {type(transmitters)}",
    ):
        transmitters.transmitters = receivers

    receivers.transmitters = transmitters

    with pytest.raises(TypeError, match="Input 'loop_radius' must be of type 'float'"):
        receivers.loop_radius = "123"

    receivers.loop_radius = 123.0

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
    new_workspace = Workspace.create(Path(tmp_path) / r"testGFEM_copy.geoh5")
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
    new_workspace = Workspace.create(Path(tmp_path) / r"testGFEM_copy2.geoh5")
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
