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
from pathlib import Path

from geoh5py.objects import BaseEMSurvey
from geoh5py.workspace import Workspace


def test_create_survey_tem():

    with tempfile.TemporaryDirectory() as tempdir:
        path = Path(tempdir) / r"testATEM.geoh5"

        # Create a workspace
        workspace = Workspace(path)
        survey = BaseEMSurvey(workspace)

        for attr in [
            "default_input_types",
            "default_units",
            "receivers",
            "transmitters",
            "unit",
            "survey_type",
        ]:
            assert (
                getattr(survey, attr, None) is None
            ), f"Attribute {attr} of the BaseEMSurvey should be None."
