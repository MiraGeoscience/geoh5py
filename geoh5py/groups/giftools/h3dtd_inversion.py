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
from geoh5py.groups.giftools.inversion_base import (
    ASSIGN_CON_RES_FIELD,
    BOUND_MODEL_LOWER_FIELD,
    INVERSION_CONTROL_PARAMETERS,
)


# Local alias to save visual space and make the parameters dict a bit easier to read.
BASE = BASE_PARAMETERS
CONTROL = INVERSION_CONTROL_PARAMETERS


# Fields unique to H3DTDinv

H3DTDINV_PARAMETERS: dict[str, Any] = {
    "assignConRes": ASSIGN_CON_RES_FIELD,
    "beta_given": CONTROL["beta_given"],
    "beta_one": merge_field(CONTROL["beta_one"], value=1e-9),
    "beta_two": merge_field(CONTROL["beta_two"], value=1.0e2),
    "cell_weight": BASE["cell_weight"],
    "delta_beta": merge_field(CONTROL["delta_beta"], value=2.0e-1),
    "face_weight": BASE["face_weight"],
    "gn_tolerance": merge_field(CONTROL["gn_tolerance"], visible=False),
    "initial_model": merge_field(BASE["initial_model"], default=1e-2, value=1e-2),
    "inversion_chifact": CONTROL["inversion_chifact"],
    "ipcg_iterations": CONTROL["ipcg_iterations"],
    "ipcg_tolerance": CONTROL["ipcg_tolerance"],
    "iterations_per_beta": CONTROL["iterations_per_beta"],
    "length_scales": merge_field(BASE["length_scales"], alpha_s=1e-3),
    "matlab": "H3DTDinversion",
    "mesh": BASE["mesh"],
    "model_perturbation": merge_field(CONTROL["model_perturbation"], visible=False),
    "reference_model": merge_field(BASE["reference_model"], default=1e-2, value=1e-2),
    "results_loaded": BASE["results_loaded"],
    "rx_data": {
        "gifType": "TEMdata",
        "label": "Data",
        "main": True,
        "meshType": "",
        "value": "",
    },
    "rx_data_dec": {
        "label": "Magnetic declination",
        "lineEdit": False,
        "main": True,
        "max": 360.0,
        "min": -360.0,
        "precision": 2,
        "value": 0.0,
        "visible": False,
    },
    "rx_data_inc": {
        "label": "Magnetic inclination",
        "lineEdit": False,
        "main": True,
        "max": 90.0,
        "min": -90.0,
        "precision": 2,
        "value": 0.0,
        "visible": False,
    },
    "smooth_mod": BASE["smooth_mod"],
    "solver": {
        "choiceList": ["pardiso", "mumps"],
        "group": "Inversion parameters",
        "label": "Solver type",
        "value": "pardiso",
        "visible": False,
    },
    "topography": merge_field(BASE["topography"], drop_keys=("enabled", "optional")),
    "update_ref": CONTROL["update_ref"],
    "uuid": BASE["uuid"],
    "version": "1",
    "working_directory": BASE["working_directory"],
    "write_to_disk": {
        "group": "Inversion parameters",
        "label": "Write factorizations to disk",
        "tooltip": "Alternatively store in RAM",
        "value": False,
        "visible": False,
    },
    "xbounds_defined": {
        "group": "Model parameters",
        "label": "No bounds",
        "value": True,
    },
    "xlower_model": merge_field(
        BOUND_MODEL_LOWER_FIELD,
        default=1e-7,
        dependency="xbounds_defined",
        dependencyType="disabled",
        label="Lower bounds",
        value=1e-7,
    ),
    "xupper_model": merge_field(
        BASE["bound_model_upper"],
        default=1.0e5,
        dependency="xbounds_defined",
        dependencyType="disabled",
        label="Upper bounds",
        value=1.0e5,
    ),
    "xy_localize": BASE["xy_localize"],
}


class H3DTDInv(BaseGIFtoolsGroup):
    """Inversion group for UBC-H3DTDInv."""

    _TYPE_UID = UUID("{4f864000-15a1-4381-afec-b274ab765568}")
    _default_name = "H3DTDinv"
    _default_parameters: dict[str, Any] = H3DTDINV_PARAMETERS
