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

from geoh5py.groups.giftools.parameters import (
    BASE_LENGTH_SCALES,
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
    "auto_threshold": {
        "choiceList": ["Relative error", "Specify value"],
        "default": "Relative error",
        "enabled": True,
        "group": "Wavelet compression",
        "groupOptional": True,
        "label": "Threshold mode",
        "value": "Relative error",
    },
    "bound_model_upper": {
        "association": "Cell",
        "dataType": "Float",
        "default": 10.0,
        "group": "Model parameters",
        "isValue": True,
        "label": "Bounds (upper)",
        "main": False,
        "parent": "mesh",
        "property": "",
        "value": 10.0,
    },
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
    "inversion_mode": {
        "choiceList": ["Chifactor", "Single beta"],
        "default": "Chifactor",
        "group": "Inversion stopping criteria",
        "label": "Inversion mode",
        "value": "Chifactor",
    },
    "inversion_par": {
        "default": 1.0,
        "group": "Inversion stopping criteria",
        "label": "Chifactor or beta",
        "min": 1e-12,
        "value": 1.0,
    },
    "inversion_tolerance": {
        "default": 5.0e-2,
        "group": "Inversion stopping criteria",
        "label": "Tolerance",
        "max": 0.5,
        "min": 1e-3,
        "value": 5.0e-2,
    },
    "length_scales": merge_field(BASE_LENGTH_SCALES, is_length=True),
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
    "matrix_file": {
        "enabled": False,
        "fileDescription": ["UBC matrix file"],
        "fileType": ["mtx"],
        "label": "Sensitivity file",
        "optional": True,
        "value": "",
    },
    "mesh": {
        "default": "",
        "label": "Mesh",
        "main": True,
        "meshType": "{b020a277-90e2-4cd7-84d6-612ee3f25051}",
        "value": "",
    },
    "reference_model": {
        "association": "Cell",
        "dataType": "Float",
        "default": 1.0e-8,
        "group": "Model parameters",
        "isValue": True,
        "label": "Reference model",
        "main": False,
        "parent": "mesh",
        "property": "",
        "value": 1.0e-8,
    },
    "results_loaded": BASE_PARAMETERS["results_loaded"],
    "smooth_mod": BASE_PARAMETERS["smooth_mod"],
    "threshold": {
        "default": 5.0e-2,
        "enabled": True,
        "group": "Wavelet compression",
        "groupOptional": True,
        "label": "Threshold value",
        "value": 5.0e-2,
    },
    "topography": {
        "default": "",
        "enabled": False,
        "label": "Topography",
        "main": True,
        "meshType": "{f26feba3-aded-494b-b9e9-b2bbcbe298e1}",
        "optional": True,
        "value": "",
    },
    "version": "6",
    "wavelet": {
        "choiceList": [
            "daub1",
            "daub2",
            "daub3",
            "daub4",
            "daub5",
            "daub6",
            "symm4",
            "symm5",
            "symm6",
        ],
        "enabled": True,
        "group": "Wavelet compression",
        "groupOptional": True,
        "label": "Wavelet type",
        "value": "daub2",
    },
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
    "xy_localize": {
        "default": False,
        "label": "Localize coordinates",
        "main": True,
        "tooltip": "Writes files to disk with respect to UBC origin of 3D grid",
        "value": False,
    },
}

# base "bound_model_lower" field that can be easily edited for any specific differences,
# e.g. different default and value or using "dataGroupType" instead of "dataType"
BOUND_MODEL_LOWER_FIELD: dict[str, Any] = {
    "association": "Cell",
    "dataType": "Float",
    "default": -10.0,
    "group": "Model parameters",
    "isValue": True,
    "label": "Bounds (lower)",
    "main": False,
    "parent": "mesh",
    "property": "",
    "value": -10.0,
}
