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
from uuid import UUID

from geoh5py.groups.giftools.base import BaseGIFtoolsGroup


# Default parameters for the maginv3d_60 inversion group.
# Note -- as we add support for more GIFtools groups, these may be able to be restructured to avoid
# some redundancy between groups.

# pylint: disable=duplicate-code
MAGINV3D_PARAMETERS: dict[str, Any] = {
    "Lp_model": {
        "association": "Cell",
        "dataType": "Float",
        "default": 2,
        "group": "Blocky model norms",
        "isValue": True,
        "label": "Amplitude (A)",
        "main": False,
        "parent": "mesh",
        "property": "",
        "value": 2,
    },
    "Lqx_model": {
        "association": "Cell",
        "dataType": "Float",
        "default": 2,
        "group": "Blocky model norms",
        "isValue": True,
        "label": "Easting derivative (E)",
        "main": False,
        "parent": "mesh",
        "property": "",
        "value": 2,
    },
    "Lqy_model": {
        "association": "Cell",
        "dataType": "Float",
        "default": 2,
        "group": "Blocky model norms",
        "isValue": True,
        "label": "Northing derivative (N)",
        "main": False,
        "parent": "mesh",
        "property": "",
        "value": 2,
    },
    "Lqz_model": {
        "association": "Cell",
        "dataType": "Float",
        "default": 2,
        "group": "Blocky model norms",
        "isValue": True,
        "label": "Vertical derivative (Z)",
        "main": False,
        "parent": "mesh",
        "property": "",
        "value": 2,
    },
    "active_model": {
        "association": "Cell",
        "dataType": ["Integer", "Boolean"],
        "default": "",
        "enabled": False,
        "group": "Model parameters",
        "label": "Active model",
        "main": False,
        "optional": True,
        "parent": "mesh",
        "suffix": ".act",
        "value": "",
    },
    "auto_threshold": {
        "choiceList": ["Relative error", "Specify value"],
        "default": "Relative error",
        "enabled": True,
        "group": "Wavelet compression",
        "groupOptional": True,
        "label": "Threshold mode",
        "value": "Relative error",
    },
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
    "bound_model_upper": {
        "association": "Cell",
        "dataType": "Float",
        "default": 10,
        "group": "Model parameters",
        "isValue": True,
        "label": "Bounds (upper)",
        "main": False,
        "parent": "mesh",
        "property": "",
        "value": 10,
    },
    "cell_weight": {
        "association": "Cell",
        "dataType": "Float",
        "default": "",
        "enabled": False,
        "group": "Model objective function",
        "label": "Weights (Ws)",
        "ndv": 1,
        "optional": True,
        "parent": "mesh",
        "suffix": ".wgt",
        "value": "",
    },
    "data": {
        "default": "",
        "gifType": ["MAGdata", "MAGAMPdata"],
        "label": "Data",
        "main": True,
        "meshType": "",
        "value": "",
    },
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
        "min": 0,
        "value": 3,
    },
    "depth_weighting_z0": {
        "default": -1234567,
        "dependency": "default_decay",
        "group": "Sensitivity weight options",
        "label": "Weighting offset (m)",
        "min": 0,
        "value": 0,
    },
    "face_weight": {
        "association": "Face",
        "dataType": "Float",
        "default": "",
        "enabled": False,
        "group": "Model objective function",
        "label": "Weights (Wxyz)",
        "ndv": 1,
        "optional": True,
        "parent": "mesh",
        "suffix": ".wgt",
        "value": "",
    },
    "initial_model": {
        "association": "Cell",
        "dataType": "Float",
        "default": 0.0010000000474974513,
        "group": "Model parameters",
        "isValue": True,
        "label": "Initial model",
        "main": False,
        "parent": "mesh",
        "property": "",
        "value": 0.0010000000474974513,
    },
    "inversion_mode": {
        "choiceList": ["Chifactor", "Single beta"],
        "default": "Chifactor",
        "group": "Inversion stopping criteria",
        "label": "Inversion mode",
        "value": "Chifactor",
    },
    "inversion_par": {
        "default": 1,
        "group": "Inversion stopping criteria",
        "label": "Chifactor or beta",
        "min": 9.999999960041972e-13,
        "value": 1,
    },
    "inversion_tolerance": {
        "default": 0.05000000074505806,
        "group": "Inversion stopping criteria",
        "label": "Tolerance",
        "max": 0.5,
        "min": 0.0010000000474974513,
        "value": 0.05000000074505806,
    },
    "length_scales": {
        "alpha_s": 0.0010000000474974513,
        "alpha_x": 1,
        "alpha_y": 1,
        "alpha_z": 1,
        "group": "Model objective function",
        "is_length": True,
        "length_x": 100,
        "length_y": 100,
        "length_z": 100,
        "parent": "mesh",
    },
    "lp_epsilon": {
        "default": -1234567,
        "enabled": False,
        "group": "Block model scaling",
        "groupOptional": True,
        "label": "A epsilon",
        "value": 0.0010000000474974513,
    },
    "lp_lq_scale": {
        "default": -1234567,
        "enabled": False,
        "group": "Block model scaling",
        "groupOptional": True,
        "label": "Scaling of A vs ENZ",
        "value": 1,
    },
    "lq_epsilon": {
        "default": -1234567,
        "enabled": False,
        "group": "Block model scaling",
        "groupOptional": True,
        "label": "ENZ epsilon",
        "value": 9.999999747378752e-06,
    },
    "matlab": "MAGinversion",
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
        "default": 0,
        "group": "Model parameters",
        "isValue": True,
        "label": "Reference model",
        "main": False,
        "parent": "mesh",
        "property": "",
        "value": 0,
    },
    "results_loaded": False,
    "smooth_mod": {
        "default": False,
        "group": "Model objective function",
        "label": "Reference model in Wxyz",
        "tooltip": "Wxyz(m-mref): SMOOTH_MOD_DIF option",
        "value": False,
    },
    "threshold": {
        "default": 0.05000000074505806,
        "enabled": True,
        "group": "Wavelet compression",
        "groupOptional": True,
        "label": "Threshold value",
        "value": 0.05000000074505806,
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
    "uuid": "",
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
    "working_directory": "",
    "xy_localize": {
        "default": False,
        "label": "Localize coordinates",
        "main": True,
        "tooltip": "Writes files to disk with respect to UBC origin of 3D grid",
        "value": False,
    },
}


class MagInv3D(BaseGIFtoolsGroup):
    """Inversion group for UBC-MAGINV3D (maginv3d_60)."""

    _TYPE_UID = UUID("{b99e8db8-e118-4042-864e-9e1128f2d1e6}")
    _default_name = "maginv3d_60"
    _default_parameters: dict[str, Any] = MAGINV3D_PARAMETERS
