# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2026 Mira Geoscience Ltd.                                     '
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

from typing import Any
from uuid import UUID

from geoh5py.groups.giftools.base import BaseGIFtoolsGroup
from geoh5py.groups.giftools.parameters import BASE_PARAMETERS, merge_field
from geoh5py.groups.giftools.potential_field_base import (
    BOUND_MODEL_LOWER_FIELD,
    POTENTIAL_FIELD_PARAMETERS,
)


# Fields unique to gzinv3d_60, in addition to the shared potential field parameters.

GZINV3D_PARAMETERS: dict[str, Any] = {
    **POTENTIAL_FIELD_PARAMETERS,
    "bound_model_lower": merge_field(BOUND_MODEL_LOWER_FIELD),
    "data": {
        "default": "",
        "gifType": ["GRAVdata"],
        "label": "Data",
        "main": True,
        "meshType": "",
        "value": "",
    },
    "matlab": "GRAVinversion",
    "uuid": BASE_PARAMETERS["uuid"],
}


class GZInv3D(BaseGIFtoolsGroup):
    """Inversion group for UBC-GZINV3D (gzinv3d_60)."""

    _TYPE_UID = UUID("{20eb4ff8-bdfe-43f3-8745-f418dcc9e14a}")
    _default_name = "gzinv3d_60"
    _default_parameters: dict[str, Any] = GZINV3D_PARAMETERS
