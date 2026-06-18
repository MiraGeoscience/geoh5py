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

import copy
from typing import Any
from uuid import UUID

from geoh5py.groups.giftools.base import BaseGIFtoolsGroup
from geoh5py.groups.giftools.parameters import BASE_PARAMETERS
from geoh5py.groups.giftools.potential_field_base import POTENTIAL_FIELD_PARAMETERS


# Fields unique to maginv3d, in addition to the shared potential field parameters.

MAGINV3D_PARAMETERS: dict[str, Any] = {
    **copy.deepcopy(POTENTIAL_FIELD_PARAMETERS),
    "bound_model_lower": {
        "association": "Cell",
        "dataType": "Float",
        "default": 0,
        "group": "Model parameters",
        "isValue": True,
        "label": "Bounds (lower)",
        "main": False,
        "parent": "mesh",
        "property": "",
        "value": 0,
    },
    "data": {
        "default": "",
        "gifType": ["MAGdata", "MAGAMPdata"],
        "label": "Data",
        "main": True,
        "meshType": "",
        "value": "",
    },
    "matlab": "MAGinversion",
    "uuid": BASE_PARAMETERS["uuid"],
}


class MagInv3D(BaseGIFtoolsGroup):
    """Inversion group for UBC-MAGINV3D (maginv3d_60)."""

    _TYPE_UID = UUID("{b99e8db8-e118-4042-864e-9e1128f2d1e6}")
    _default_name = "maginv3d_60"
    _default_parameters: dict[str, Any] = MAGINV3D_PARAMETERS
