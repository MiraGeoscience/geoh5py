#  Copyright (c) 2023 Mira Geoscience Ltd Ltd.
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

from geoh5py.objects.surveys.electromagnetics.base import BaseEMSurvey
from geoh5py.workspace import Workspace


def test_create_survey_tem(tmp_path):
    h5file_path = tmp_path / r"testATEM.geoh5"

    with Workspace(h5file_path) as workspace:
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
