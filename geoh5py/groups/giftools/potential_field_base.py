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

from typing import Any

from geoh5py.groups.giftools.base import (
    BASE_PARAMETERS,
    merge_field,
)


# Parameters shared by GIFtools potential-field inversion groups that operate on a
# block (tensor) mesh, such as maginv3d_60 and gginv3d.

POTENTIAL_FIELD_PARAMETERS: dict[str, Any] = {
    "Lp_model": {
        "association": "Cell",
        "dataType": "Float",
        "default": 2.0,
        "group": "Blocky model norms",
        "isValue": True,
        "label": "Amplitude (A)",
        "main": False,
        "parent": "mesh",
        "property": "",
        "value": 2.0,
    },
    "Lqx_model": {
        "association": "Cell",
        "dataType": "Float",
        "default": 2.0,
        "group": "Blocky model norms",
        "isValue": True,
        "label": "Easting derivative (E)",
        "main": False,
        "parent": "mesh",
        "property": "",
        "value": 2.0,
    },
    "Lqy_model": {
        "association": "Cell",
        "dataType": "Float",
        "default": 2.0,
        "group": "Blocky model norms",
        "isValue": True,
        "label": "Northing derivative (N)",
        "main": False,
        "parent": "mesh",
        "property": "",
        "value": 2.0,
    },
    "Lqz_model": {
        "association": "Cell",
        "dataType": "Float",
        "default": 2.0,
        "group": "Blocky model norms",
        "isValue": True,
        "label": "Vertical derivative (Z)",
        "main": False,
        "parent": "mesh",
        "property": "",
        "value": 2.0,
    },
    # add main=False relative to the universal base.
    "active_model": merge_field(BASE_PARAMETERS["active_model"], main=False),
    "auto_threshold": BASE_PARAMETERS["auto_threshold"],
    "bound_model_upper": BASE_PARAMETERS["bound_model_upper"],
    "cell_weight": BASE_PARAMETERS["cell_weight"],
    "default_decay": {
        "default": False,
        "group": "Sensitivity weight options",
        "label": "Specify exponent decay",
        "value": False,
    },
    "depth_weighting": {
        "choiceList": ["Depth", "Distance"],
        "default": "Depth",
        "group": "Sensitivity weight options",
        "groupDependency": "matrix_file",
        "groupDependencyType": "disabled",
        "label": "Weight type",
        "value": "Distance",
    },
    "depth_weighting_beta": {
        "default": -1234567,
        "dependency": "default_decay",
        "group": "Sensitivity weight options",
        "label": "Weighting decay constant",
        "min": 1.0e-8,
        "value": 3.0,
    },
    "depth_weighting_z0": {
        "default": -1234567,
        "dependency": "default_decay",
        "group": "Sensitivity weight options",
        "label": "Weighting offset (m)",
        "min": 1.0e-8,
        "value": 1.0e-8,
    },
    "face_weight": BASE_PARAMETERS["face_weight"],
    "initial_model": BASE_PARAMETERS["initial_model"],
    "inversion_mode": BASE_PARAMETERS["inversion_mode"],
    "inversion_par": BASE_PARAMETERS["inversion_par"],
    "inversion_tolerance": BASE_PARAMETERS["inversion_tolerance"],
    "length_scales": BASE_PARAMETERS["length_scales"],
    "lp_epsilon": {
        "default": -1234567,
        "enabled": False,
        "group": "Block model scaling",
        "groupOptional": True,
        "label": "A epsilon",
        "value": 1e-3,
    },
    "lp_lq_scale": {
        "default": -1234567,
        "enabled": False,
        "group": "Block model scaling",
        "groupOptional": True,
        "label": "Scaling of A vs ENZ",
        "value": 1.0,
    },
    "lq_epsilon": {
        "default": -1234567,
        "enabled": False,
        "group": "Block model scaling",
        "groupOptional": True,
        "label": "ENZ epsilon",
        "value": 1e-5,
    },
    "matrix_file": BASE_PARAMETERS["matrix_file"],
    "mesh": BASE_PARAMETERS["mesh"],
    "reference_model": BASE_PARAMETERS["reference_model"],
    "results_loaded": BASE_PARAMETERS["results_loaded"],
    "smooth_mod": BASE_PARAMETERS["smooth_mod"],
    "threshold": BASE_PARAMETERS["threshold"],
    "topography": BASE_PARAMETERS["topography"],
    "version": "6",
    "wavelet": BASE_PARAMETERS["wavelet"],
    "wavelet_diagnostics": {
        "default": False,
        "group": "Wavelet compression",
        "groupDependency": "matrix_file",
        "groupDependencyType": "disabled",
        "groupOptional": True,
        "label": "Output diagnostics files",
        "main": False,
        "value": False,
    },
    "working_directory": BASE_PARAMETERS["working_directory"],
    "xy_localize": BASE_PARAMETERS["xy_localize"],
}
