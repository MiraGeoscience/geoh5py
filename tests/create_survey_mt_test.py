#  Copyright (c) 2021 Mira Geoscience Ltd.
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

from geoh5py.objects import Magnetotellurics
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
        mt_survey = Magnetotellurics.create(workspace, vertices=vertices, name=name)

        with pytest.raises(TypeError):
            mt_survey.metadata = "Hello World"

        with pytest.raises(KeyError):
            mt_survey.metadata = {"Hello World": {}}

        with pytest.raises(KeyError):
            mt_survey.metadata = {"EM Dataset": {}}

        with pytest.raises(TypeError):
            mt_survey.channels = 1.0

        with pytest.raises(AttributeError):
            mt_survey.add_frequency_data(123.0)

        mt_survey.channels = [5.0, 10.0, 100.0]

        with pytest.raises(TypeError):
            mt_survey.add_frequency_data(123.0)

        # Create some simple data
        data = {}
        d_vec = (x_loc * y_loc).ravel() / 200.0
        for c_ind, component in enumerate(
            ["Zxx (real)", "Zxx (imaginary)", "Zxy (real)", "Zyy (real)"]
        ):
            comp_dict = {}

            for f_ind, freq in enumerate(mt_survey.channels):
                values = (c_ind + 1.0) * np.sin(f_ind * np.pi * d_vec)
                comp_dict[freq] = {"values": values}

            if c_ind == 0:
                with pytest.raises(TypeError):
                    mt_survey.add_frequency_data({component: values})

                with pytest.raises(ValueError):
                    mt_survey.add_frequency_data(
                        {component: {ind: values for ind in range(2)}}
                    )

                with pytest.raises(KeyError):
                    mt_survey.add_frequency_data(
                        {component: {ind: values for ind in range(3)}}
                    )

                with pytest.raises(TypeError):
                    mt_survey.add_frequency_data(
                        {component: {freq: values for freq in mt_survey.channels}}
                    )

            data[component] = comp_dict

        mt_survey.add_frequency_data(data)

        assert len(mt_survey.metadata["EM Dataset"]["Property groups"]) == len(
            mt_survey.property_groups
        ), "Metadata 'Property groups' malformed"

        workspace.finalize()

        # Re-open the workspace and read data back in
        new_workspace = Workspace(path)
        mt_survey_rec = new_workspace.get_entity(name)[0]

        # Check entities
        compare_entities(
            mt_survey, mt_survey_rec, ignore=["_parent", "_property_groups"]
        )
