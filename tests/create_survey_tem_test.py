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

import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest

from geoh5py.objects import AirborneTEMReceivers, AirborneTEMTransmitters
from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace


def test_create_survey_tem():

    name = "Survey"
    with tempfile.TemporaryDirectory() as tempdir:
        path = Path(tempdir) / r"testATEM.geoh5"

        # Create a workspace
        workspace = Workspace(path)
        xlocs = np.linspace(-1000, 1000, 10)
        vertices = np.c_[xlocs, np.random.randn(xlocs.shape[0], 2)]
        # obj = workspace.get_entity("MA1_MW_ShC_Bz_20191127_rot330_zone_e receivers")[0]
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
        receivers.transmitters = transmitters

        with pytest.raises(TypeError) as error:
            receivers.channels = {"abc": 1, "dfg": 2}

        assert f"Values provided as 'channels' must be a list of {float}" in str(
            error
        ), "Missed raising error on 'channels' change."

        receivers.channels = np.logspace(-3, -2, 10)

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

        receivers.waveform = np.c_[np.logspace(-5, -1.9, 10), np.linspace(0, 1, 10)]

        with pytest.raises(ValueError) as error:
            receivers.timing_mark = "abc"

        assert "Input timing_mark must be a float." in str(
            error
        ), "Missed raising error on type of 'timing_mark'."
        receivers.timing_mark = 10 ** -3.1

        assert (
            receivers.metadata == transmitters.metadata
        ), "Error synchronizing the transmitters and receivers metadata."

        for key in ["pitch", "roll", "yaw"]:
            with pytest.raises(TypeError) as error:
                setattr(receivers, key, "abc")

            assert f"Input '{key}' must be one of type float or uuid.UUID" in str(
                error
            ), f"Missed raising error on type of '{key}'."

            setattr(receivers, key, uuid.uuid4())
            assert (
                f"{key.capitalize()} property" in receivers.metadata["EM Dataset"]
            ), f"Wrong metadata label set on '{key}' for input uuid."
            assert (
                f"{key.capitalize()} value" not in receivers.metadata["EM Dataset"]
            ), f"Failed in removing '{key}' value from metadata."
            setattr(receivers, key, 3.0)
            assert (
                f"{key.capitalize()} value" in receivers.metadata["EM Dataset"]
            ), f"Wrong metadata label set on '{key}' for input uuid."
            assert (
                f"{key.capitalize()} property" not in receivers.metadata["EM Dataset"]
            ), f"Failed in removing '{key}' property from metadata."

        workspace.finalize()

        new_workspace = Workspace(path)
        transmitters_rec = new_workspace.get_entity(name + "_tx")[0]
        receivers_rec = new_workspace.get_entity(name + "_rx")[0]

        # Check entities
        compare_entities(
            transmitters, transmitters_rec, ignore=["_receivers", "_parent"]
        )
        compare_entities(receivers, receivers_rec, ignore=["_transmitters", "_parent"])
