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


# pylint: disable=R0914
# mypy: ignore-errors


from __future__ import annotations

import logging
import re

import numpy as np
import pytest

from geoh5py.groups import PropertyGroup
from geoh5py.objects import AirborneTEMTransmitters, MTReceivers
from geoh5py.shared.utils import compare_entities
from geoh5py.workspace import Workspace


def test_create_survey_mt(tmp_path, caplog):
    name = "TestMT"
    h5file_path = tmp_path / r"testMT.geoh5"

    with Workspace.create(h5file_path) as workspace:
        # Define the receiver locations on a grid
        x_loc, y_loc = np.meshgrid(np.linspace(-5, 5, 2), np.linspace(0.0, 20.0, 9))
        vertices = np.c_[x_loc.ravel(), y_loc.ravel(), np.zeros_like(x_loc).ravel()]

        # Create the survey from vertices
        mt_survey = MTReceivers.create(workspace, vertices=vertices, name=name)

        with pytest.raises(
            TypeError,
            match=(
                f"Provided receivers must be of type {mt_survey.default_receiver_type}. "
                f"{str} provided."
            ),
        ):
            mt_survey.receivers = "123"

        with pytest.raises(ValueError, match="Mask must be an array of shape"):
            mt_survey.copy(mask=np.r_[1, 2, 3])

        for key, value in {
            "input_type": "Rx only",
            "survey_type": "Magnetotellurics",
            "unit": "Hertz (Hz)",
        }.items():
            assert getattr(mt_survey, key) == value, f"Error setting defaults for {key}"

        with pytest.raises(ValueError, match="Input 'input_type' must be one of"):
            mt_survey.input_type = "XYZ"

        mt_survey.input_type = "Rx only"

        with pytest.raises(TypeError, match="'metadata' must be of type 'dict'"):
            mt_survey.metadata = "Hello World"

        metadata = mt_survey.default_metadata.copy()
        metadata["EM Dataset"].pop("Unit")
        with caplog.at_level(logging.WARNING):
            mt_survey.metadata = metadata

        assert "Unit" in caplog.text

        with pytest.raises(
            TypeError, match="Values provided as 'channels' must be a list"
        ):
            mt_survey.channels = 1.0

        with pytest.raises(
            AttributeError,
            match=(
                "The 'channels' attribute of an EMSurvey class must be set before "
                "the 'add_components_data' method can be used."
            ),
        ):
            mt_survey.add_components_data(123.0)

        mt_survey.channels = [5.0, 10.0, 100.0]

        with pytest.raises(TypeError, match="Input data must be nested dictionaries"):
            mt_survey.add_components_data(123.0)

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
                with pytest.raises(
                    TypeError,
                    match=re.escape(
                        "List of values provided for component 'Zxx (real)' "
                        "must be a list of "
                    ),
                ):
                    mt_survey.add_components_data({component: values})

                with pytest.raises(
                    TypeError,
                    match=(
                        "Given value to data 5.0 should of type "
                        "<class 'dict'> or attributes"
                    ),
                ):
                    mt_survey.add_components_data(
                        {component: dict.fromkeys(mt_survey.channels, values)}
                    )

            # Give well-formed dictionary
            data[component] = comp_dict

        mt_survey.add_components_data(data)

        assert len(mt_survey.metadata["EM Dataset"]["Property groups"]) == len(
            mt_survey.property_groups
        ), "Metadata 'Property groups' malformed"

        with pytest.raises(
            AttributeError,
            match=f"The 'transmitters' attribute cannot be set on class {type(mt_survey)}.",
        ):
            mt_survey.transmitters = AirborneTEMTransmitters

        # Re-open the workspace and read data back in
        new_workspace = Workspace(h5file_path)
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

        with Workspace.create(tmp_path / r"testMT_copy.geoh5") as copy_workspace:
            mt_survey_rec.copy(copy_workspace)
            mt_survey_extent = mt_survey_rec.copy_from_extent(
                np.vstack([[-6.0, -1.0], [6.0, 6.0]]), parent=copy_workspace
            )

            assert mt_survey_extent.n_vertices == 6
            for child_a, child_b in zip(
                [
                    child
                    for child in mt_survey_extent.children
                    if not isinstance(child, PropertyGroup)
                ],
                [
                    child
                    for child in mt_survey_rec.children
                    if not isinstance(child, PropertyGroup)
                ],
                strict=False,
            ):
                np.testing.assert_array_almost_equal(child_a.values, child_b.values[:6])
