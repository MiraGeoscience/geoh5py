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

# pylint: disable=R0914

import tempfile
from pathlib import Path

import numpy as np
import pytest

from geoh5py.objects import AirborneTEMTransmitters, MTReceivers
from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace


def test_create_survey_mt():

    name = "TestMT"
    n_data = 12

    with tempfile.TemporaryDirectory() as tempdir:
        path = Path(tempdir) / r"testMT.geoh5"

        # Create a workspace
        workspace = Workspace(path)

        # Create sources along line
        x_loc, y_loc = np.meshgrid(np.arange(n_data), np.arange(-1, 3))
        vertices = np.c_[x_loc.ravel(), y_loc.ravel(), np.zeros_like(x_loc).ravel()]

        # Define the receiver locations on 2 lines, 60 m apart
        x_loc, y_loc = np.meshgrid(np.linspace(-5, 5, 2), np.linspace(0.0, 20.0, 9))
        vertices = np.c_[x_loc.ravel(), y_loc.ravel(), np.zeros_like(x_loc).ravel()]

        # Create the survey from vertices
        mt_survey = MTReceivers.create(workspace, vertices=vertices, name=name)

        with pytest.raises(UserWarning) as error:
            mt_survey.receivers = "123"

        assert (
            "Attribute 'receivers' of the class 'MTReceivers' must reference to self."
            in str(error)
        ), "Missed raising UserWarning on setting 'receivers' on self."

        for key, value in {
            "input_type": "Rx only",
            "survey_type": "Magnetotellurics",
            "unit": "Hertz (Hz)",
        }.items():
            assert getattr(mt_survey, key) == value, f"Error setting defaults for {key}"

        with pytest.raises(ValueError) as excinfo:
            mt_survey.input_type = "XYZ"

        assert "Input 'input_type' must be one of" in str(
            excinfo
        ), "Failed to raise ValueError on input_type."

        mt_survey.input_type = "Rx only"

        with pytest.raises(TypeError) as excinfo:
            mt_survey.metadata = "Hello World"
        assert "'metadata' must be of type 'dict'" in str(excinfo)

        with pytest.raises(KeyError) as excinfo:
            mt_survey.metadata = {"EM Dataset": {}}
        assert f"{list(mt_survey.default_metadata['EM Dataset'].keys())}" in str(
            excinfo
        )

        mt_survey.metadata = mt_survey.default_metadata

        with pytest.raises(TypeError) as excinfo:
            mt_survey.channels = 1.0
        assert "Values provided as 'channels' must be a list" in str(excinfo)

        with pytest.raises(AttributeError) as excinfo:
            mt_survey.add_component_data(123.0)
        assert (
            "The 'channels' attribute of an EMSurvey class must be set before "
            "the 'add_component_data' method can be used."
        ) in str(excinfo)
        mt_survey.channels = [5.0, 10.0, 100.0]

        with pytest.raises(TypeError) as excinfo:
            mt_survey.add_component_data(123.0)
        assert "Input data must be nested dictionaries" in str(excinfo)

        # Create some simple data
        data = {}
        d_vec = (x_loc * y_loc).ravel() / 200.0
        for c_ind, component in enumerate(
            ["Zxx (real)", "Zxx (imaginary)", "Zxy (real)", "Zyy (real)"]
        ):
            comp_dict = {}
            for f_ind, freq in enumerate(mt_survey.channels):
                values = (c_ind + 1.0) * np.sin(f_ind * np.pi * d_vec)
                comp_dict[f"{component}_{freq}"] = {"values": values}

            if c_ind == 0:
                with pytest.raises(TypeError) as excinfo:
                    mt_survey.add_component_data({component: values})
                assert (
                    "List of values provided for component 'Zxx (real)' "
                    "must be a list of "
                ) in str(excinfo)

                with pytest.raises(TypeError) as excinfo:
                    mt_survey.add_component_data(
                        {component: {ind: values for ind in mt_survey.channels}}
                    )
                assert (
                    "Given value to data 5.0 should of type "
                    "<class 'dict'> or attributes"
                ) in str(excinfo)

            # Give well-formed dictionary
            data[component] = comp_dict

        mt_survey.add_component_data(data)

        assert len(mt_survey.metadata["EM Dataset"]["Property groups"]) == len(
            mt_survey.property_groups
        ), "Metadata 'Property groups' malformed"

        with pytest.raises(AttributeError) as excinfo:
            mt_survey.transmitters = AirborneTEMTransmitters

        assert "does not have transmitters." in str(
            excinfo
        ), "Failed to raise AttributeError."

        workspace.finalize()

        # Re-open the workspace and read data back in
        new_workspace = Workspace(path)
        mt_survey_rec = new_workspace.get_entity(name)[0]
        diffs = []
        for key, value in mt_survey_rec.metadata["EM Dataset"].items():
            if mt_survey.metadata["EM Dataset"][key] != value:
                diffs.append(key)
        # Check entities
        compare_entities(
            mt_survey,
            mt_survey_rec,
            ignore=["_receivers", "_parent", "_property_groups"],
        )
