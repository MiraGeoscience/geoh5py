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


# Local alias to save visual space and make the parameters dict a bit easier to read.
_PF = POTENTIAL_FIELD_PARAMETERS


# Fields unique to ipinv3d. Only a subset of POTENTIAL_FIELD_PARAMETERS is used

# *Note - this file couldn't reuse the POTENTIAL_FIELD_PARAMETERS dict as it doesn't require some of
# the fields in that one, and it needs to adjust several other fields from
# POTENTIAL_FIELD_PARAMETERS. If more GIFtools groups seem to have the same issue, it would make
# sense to look into further refactors or breaking up of large base dicts. My thinking is it's
# better to increase complexity of base dicts like BASE_PARAMETERS and POTENTIAL_FIELD_PARAMETERS
# than to have a lot of duplicate code in the derived dicts, but it depends on the patterns that
# emerge down the line.

IPINV3D_PARAMETERS: dict[str, Any] = {
    "active_model": merge_field(_PF["active_model"], visible=False),
    "auto_threshold": merge_field(_PF["auto_threshold"], groupOptional=False),
    "bound_model_lower": merge_field(
        BOUND_MODEL_LOWER_FIELD, value=1.0e-8, visible=False
    ),
    "bound_model_upper": _PF["bound_model_upper"],
    "cell_weight": _PF["cell_weight"],
    "data": {
        "default": "",
        "gifType": "IP3Ddata",
        "label": "Data",
        "main": True,
        "meshType": "",
        "value": "",
    },
    "face_weight": _PF["face_weight"],
    "forward_tolerance": {
        "default": 1e-5,
        "label": "Forward solver tolerance",
        "min": 1e-12,
        "value": 1e-5,
    },
    "initial_model": _PF["initial_model"],
    "inversion_mode": _PF["inversion_mode"],
    "inversion_par": _PF["inversion_par"],
    "inversion_tolerance": _PF["inversion_tolerance"],
    "length_scales": _PF["length_scales"],
    "matlab": "IP3Dinversion",
    "matrix_file": _PF["matrix_file"],
    "mesh": _PF["mesh"],
    "model_conductivity": {
        "association": "Cell",
        "dataType": "Float",
        "label": "Conductivity model",
        "main": True,
        "ndv": 1e-8,
        "parent": "mesh",
        "suffix": ".con",
        "tooltip": "Conductivity in S/m",
        "value": "",
    },
    "reference_model": _PF["reference_model"],
    "results_loaded": _PF["results_loaded"],
    "smooth_mod": merge_field(_PF["smooth_mod"], visible=False),
    "store_vectors": {
        "default": -1,
        "enabled": False,
        "label": "Number of vectors to store in memory",
        "optional": True,
        "value": -1,
    },
    "threshold": merge_field(_PF["threshold"], groupOptional=False),
    "topography": _PF["topography"],
    "uuid": BASE_PARAMETERS["uuid"],
    "version": "5",
    "wavelet": merge_field(_PF["wavelet"], groupOptional=False),
    "working_directory": _PF["working_directory"],
    "xy_localize": _PF["xy_localize"],
}


class IPInv3D(BaseGIFtoolsGroup):
    """Inversion group for UBC-IPINV3D (ipinv3d)."""

    _TYPE_UID = UUID("{9f9543a0-e857-4a56-ab66-9f21e2b002c6}")
    _default_name = "ipinv3d"
    _default_parameters: dict[str, Any] = IPINV3D_PARAMETERS
