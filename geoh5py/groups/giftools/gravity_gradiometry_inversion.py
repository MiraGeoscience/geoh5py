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

from geoh5py.groups.giftools.base import BASE_PARAMETERS, BaseGIFtoolsGroup, merge_field
from geoh5py.groups.giftools.potential_field_base import (
    BOUND_MODEL_LOWER_FIELD,
    POTENTIAL_FIELD_PARAMETERS,
)


# Fields unique to gginv3d, in addition to the shared potential field parameters.

GGINV3D_PARAMETERS: dict[str, Any] = {
    **POTENTIAL_FIELD_PARAMETERS,
    "bound_model_lower": merge_field(BOUND_MODEL_LOWER_FIELD),
    "data": {
        "default": "",
        "gifType": ["GGdata", "FALCONdata"],
        "label": "Data",
        "main": True,
        "meshType": "",
        "value": "",
    },
    "matlab": "GGinversion",
    "uuid": BASE_PARAMETERS["uuid"],
}


class GGInv3D(BaseGIFtoolsGroup):
    """Inversion group for UBC-GGINV3D (gginv3d)."""

    _TYPE_UID = UUID("{0f080369-b3a3-464c-83fa-9b3c1efa9895}")
    _default_name = "gginv3d"
    _default_parameters: dict[str, Any] = GGINV3D_PARAMETERS
