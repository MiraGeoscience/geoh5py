# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2020-2026 Mira Geoscience Ltd.                                     '
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
    LargeLoopGroundTEMReceivers,
    LargeLoopGroundTEMTransmitters,
    MovingLoopGroundTEMReceivers,
    MovingLoopGroundTEMTransmitters,
)
from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace


def make_large_loop_survey(workspace: Workspace):
    """
    Utility function to create a large loop survey.
    """
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

    receivers = LargeLoopGroundTEMReceivers.create(
        workspace, vertices=np.vstack(vertices)
    )

    transmitters = LargeLoopGroundTEMTransmitters.create(
        workspace,
        vertices=np.vstack(tx_loops),
        cells=np.vstack(tx_cells),
    )
    transmitters.tx_id_property = transmitters.parts + 1
    transmitters.tx_id_property.name = "Random ID"

    receivers.tx_id_property = np.hstack(tx_id)
    receivers.tx_id_property.name = "Random ID receivers"
    return receivers, transmitters


def test_create_survey_ground_tem_large_loop(
    tmp_path,
):  # pylint: disable=too-many-locals
    path = Path(tmp_path) / r"groundTEM.geoh5"

    # Create a workspace
    with Workspace.create(path) as workspace:
        receivers, transmitters = make_large_loop_survey(workspace)

        with pytest.raises(ValueError, match="Input 'input_type' must be one of"):
            receivers.input_type = "123"

        assert receivers.input_type == "Tx and Rx"

        receivers.channels = [10.0, 100.0]
        values = {}
        for component in ["bx", "bz"]:
            data = {}

            for ind, channel in enumerate(receivers.channels):
                data[f"channel[{ind}]"] = {
                    "values": np.ones(receivers.n_vertices) * channel
                }

            values[component] = data

        receivers.add_components_data(values)

        assert isinstance(receivers, LargeLoopGroundTEMReceivers), (
            "Entity type GroundTEMReceiversLargeLoop failed to create."
        )

        assert isinstance(transmitters, LargeLoopGroundTEMTransmitters), (
            "Entity type GroundTEMTransmittersLargeLoop failed to create."
        )

        with pytest.raises(
            TypeError, match=f" must be of type {LargeLoopGroundTEMTransmitters}"
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
        transmitter_uid = transmitters.tx_id_property.uid  # Remember for next test

    with Workspace(path) as workspace:
        transmitters = workspace.get_entity(transmitters.uid)[0]
        receivers = workspace.get_entity(receivers.uid)[0]

        assert transmitters.tx_id_property.uid == transmitter_uid, (
            "Failed to maintain transmitter id property."
        )

        with Workspace.create(
            Path(tmp_path) / r"testGround_copy.geoh5"
        ) as new_workspace:
            receivers_orig = receivers.copy(new_workspace)
            repeat_copy = receivers_orig.copy()
            assert (
                repeat_copy.metadata["EM Dataset"]["Tx ID property"]
                != receivers_orig.metadata["EM Dataset"]["Tx ID property"]
            )


def test_copy_from_extent(
    tmp_path,
):
    path = Path(tmp_path) / r"groundTEM.geoh5"

    # Create a workspace
    with Workspace.create(path) as workspace:
        receivers, transmitters = make_large_loop_survey(workspace)
        receivers.transmitters = transmitters

        with Workspace.create(
            Path(tmp_path) / r"testGround_copy_extent.geoh5"
        ) as new_workspace:
            transmitters_rec = receivers.transmitters.copy_from_extent(
                np.vstack([[-150, 300], [150, 600]]), parent=new_workspace
            )
            assert transmitters_rec.receivers.n_vertices == receivers.n_vertices / 2.0
            assert (
                transmitters_rec.n_vertices == receivers.transmitters.n_vertices / 2.0
            )

            assert list(
                transmitters_rec.tx_id_property.entity_type.value_map.map["Value"]
            ) == [b"Unknown", b"Loop 2"]


def test_copy_no_children(
    tmp_path,
):
    path = Path(tmp_path) / r"groundTEM.geoh5"

    # Create a workspace
    with Workspace.create(path) as workspace:
        receivers, transmitters = make_large_loop_survey(workspace)
        receivers.transmitters = transmitters

        new = receivers.copy(copy_children=False)
        assert new.tx_id_property is not None


def test_create_survey_ground_tem(tmp_path):
    name = "Survey"
    path = Path(tmp_path) / r"../testGTEM.geoh5"

    # Create a workspace
    with Workspace.create(path) as workspace:
        xlocs = np.linspace(-1000, 1000, 10)
        vertices = np.c_[xlocs, np.random.randn(xlocs.shape[0], 2)]
        receivers = MovingLoopGroundTEMReceivers.create(
            workspace, vertices=vertices, name=name + "_rx"
        )
        assert receivers.tx_id_property is None

        receivers.tx_id_property = np.arange(receivers.n_vertices)

        assert isinstance(receivers, MovingLoopGroundTEMReceivers), (
            "Entity type MovingLoopGroundTEMReceivers failed to create."
        )
        transmitters = MovingLoopGroundTEMTransmitters.create(
            workspace, vertices=vertices + 10.0, name=name + "_tx"
        )
        assert isinstance(transmitters, MovingLoopGroundTEMTransmitters), (
            "Entity type MovingLoopGroundTEMTransmitters failed to create."
        )

        with pytest.raises(
            TypeError, match=f" must be of type {MovingLoopGroundTEMTransmitters}"
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

        with pytest.raises(
            TypeError, match="Input 'loop_radius' must be of type 'float'"
        ):
            receivers.loop_radius = "123"

        receivers.loop_radius = 123.0

    with Workspace(path) as workspace:
        transmitters = workspace.get_entity(name + "_tx")[0]
        receivers = workspace.get_entity(name + "_rx")[0]

        # Test copying transmitter over through the receivers
        # Create a workspace
        new_workspace = Workspace.create(Path(tmp_path) / r"testGTEM_copy.geoh5")
        receivers_rec = receivers.copy(new_workspace)
        receivers_rec.tx_id_property
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
        new_workspace = Workspace.create(Path(tmp_path) / r"testGTEM_copy2.geoh5")
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
