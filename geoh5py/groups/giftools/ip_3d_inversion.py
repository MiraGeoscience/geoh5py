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
from geoh5py.groups.giftools.inversion_base import BOUND_MODEL_LOWER_FIELD


# Local alias to save visual space and make the parameters dict a bit easier to read.
BASE = BASE_PARAMETERS


# Fields unique to ipinv3d. Only a subset of BASE_PARAMETERS is used

IPINV3D_PARAMETERS: dict[str, Any] = {
    "active_model": merge_field(BASE["active_model"], main=False, visible=False),
    "auto_threshold": merge_field(BASE["auto_threshold"], groupOptional=False),
    "bound_model_lower": merge_field(
        BOUND_MODEL_LOWER_FIELD, value=1.0e-8, visible=False
    ),
    "bound_model_upper": BASE["bound_model_upper"],
    "cell_weight": BASE["cell_weight"],
    "data": {
        "default": "",
        "gifType": "IP3Ddata",
        "label": "Data",
        "main": True,
        "meshType": "",
        "value": "",
    },
    "face_weight": BASE["face_weight"],
    "forward_tolerance": {
        "default": 1e-5,
        "label": "Forward solver tolerance",
        "min": 1e-12,
        "value": 1e-5,
    },
    "initial_model": BASE["initial_model"],
    "inversion_mode": BASE["inversion_mode"],
    "inversion_par": BASE["inversion_par"],
    "inversion_tolerance": BASE["inversion_tolerance"],
    "length_scales": BASE["length_scales"],
    "matlab": "IP3Dinversion",
    "matrix_file": BASE["matrix_file"],
    "mesh": BASE["mesh"],
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
    "reference_model": BASE["reference_model"],
    "results_loaded": BASE["results_loaded"],
    "smooth_mod": merge_field(BASE["smooth_mod"], visible=False),
    "store_vectors": {
        "default": -1,
        "enabled": False,
        "label": "Number of vectors to store in memory",
        "optional": True,
        "value": -1,
    },
    "threshold": merge_field(BASE["threshold"], groupOptional=False),
    "topography": BASE["topography"],
    "uuid": BASE["uuid"],
    "version": "5",
    "wavelet": merge_field(BASE["wavelet"], groupOptional=False),
    "working_directory": BASE["working_directory"],
    "xy_localize": BASE["xy_localize"],
}


class IPInv3D(BaseGIFtoolsGroup):
    """Inversion group for UBC-IPINV3D (ipinv3d)."""

    _TYPE_UID = UUID("{9f9543a0-e857-4a56-ab66-9f21e2b002c6}")
    _default_name = "ipinv3d"
    _default_parameters: dict[str, Any] = IPINV3D_PARAMETERS
