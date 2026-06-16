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


# Parameters shared by the GIFtools octree inversion groups.
# Note -- this will likely be restructured as we know more about what the commonalities and
# differences are between different GIFtools/GIFtools octree/GIFtools inversion groups.
# pylint: disable=duplicate-code
OCTREE_INVERSION_PARAMETERS: dict[str, Any] = {
    "active_model": {
        "association": "Cell",
        "dataType": ["Integer", "Boolean"],
        "default": "",
        "enabled": False,
        "group": "Model parameters",
        "label": "Active model",
        "optional": True,
        "parent": "mesh",
        "suffix": ".act",
        "value": "",
    },
    "assignConRes": {
        "alternateLabel": "Resistivity",
        "group": "Model parameters",
        "label": "Model Type",
        "main": False,
        "originalLabel": "Conductivity",
        "tooltip": "Resistivity (Ohm-m) or Conductivity (S/m)",
        "value": "Conductivity",
    },
    "beta_given": {
        "default": False,
        "group": "Inversion parameters",
        "label": "Specify Beta range",
        "value": False,
    },
    "beta_one": {
        "default": -1234567,
        "dependency": "beta_given",
        "group": "Inversion parameters",
        "label": "Beta minimum",
        "min": 1e-10,
        "value": 0.0010000000474974513,
    },
    "beta_two": {
        "default": -1234567,
        "dependency": "beta_given",
        "group": "Inversion parameters",
        "label": "Beta maximum",
        "min": 1e-10,
        "value": 1000,
    },
    "bounds_defined": {
        "group": "Model parameters",
        "label": "No bounds",
        "value": True,
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
    "delta_beta": {
        "default": -1234567,
        "dependency": "beta_given",
        "group": "Inversion parameters",
        "label": "Beta step",
        "value": 0.25,
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
    "global_weight": {
        "association": "Cell",
        "dataType": "Float",
        "default": "",
        "enabled": False,
        "group": "Model objective function",
        "label": "Global weights (cell-centred)",
        "optional": True,
        "parent": "mesh",
        "value": "",
        "visible": True,
    },
    "gn_tolerance": {
        "default": 0.009999999776482582,
        "group": "Gauss-Newton options",
        "label": "Solver tolerance",
        "value": 0.009999999776482582,
    },
    "initial_model": {
        "association": "Cell",
        "dataType": "Float",
        "default": 0.0010000000474974513,
        "group": "Model parameters",
        "isValue": True,
        "label": "Initial model",
        "main": False,
        "max": 100000000,
        "min": 9.99999993922529e-09,
        "parent": "mesh",
        "property": "",
        "value": 0.0010000000474974513,
    },
    "inversion_chifact": {
        "default": 1,
        "group": "Inversion parameters",
        "label": "Chi factor",
        "value": 1,
    },
    "ipcg_iterations": {
        "default": 20,
        "group": "Gauss-Newton options",
        "label": "IPCG iterations",
        "min": 1,
        "tooltip": "Maximum iterations of incomplete-preconditioned-conjugate gradient (IPCG)",
        "value": 20,
    },
    "ipcg_tolerance": {
        "default": 0.009999999776482582,
        "group": "Gauss-Newton options",
        "label": "IPCG tolerance",
        "tooltip": "Fractional percent norm of the iterative solver residual",
        "value": 0.009999999776482582,
    },
    "iterations_per_beta": {
        "default": 3,
        "group": "Gauss-Newton options",
        "label": "Iterations per beta",
        "min": 1,
        "value": 3,
    },
    "iterative_solver": {
        "choiceList": ["Direct", "Iterative"],
        "group": "Inversion parameters",
        "label": "Solver type",
        "value": "Iterative",
        "visible": False,
    },
    "length_scales": {
        "alpha_s": 9.999999747378752e-05,
        "alpha_x": 1,
        "alpha_y": 1,
        "alpha_z": 1,
        "group": "Model objective function",
        "is_length": False,
        "length_x": 100,
        "length_y": 100,
        "length_z": 100,
        "parent": "mesh",
    },
    "lower_bound_model": {
        "association": "Cell",
        "dataType": "Float",
        "default": 0,
        "dependency": "bounds_defined",
        "dependencyType": "disabled",
        "group": "Model parameters",
        "isValue": True,
        "label": "Lower bounds",
        "main": False,
        "max": 100000000,
        "min": 9.99999993922529e-09,
        "parent": "mesh",
        "property": "",
        "value": 0,
    },
    "matlab": "",
    "mesh": {
        "default": "",
        "label": "Octree mesh",
        "main": True,
        "meshType": "{4ea87376-3ece-438b-bf12-3479733ded46}",
        "value": "",
    },
    "model_perturbation": {
        "default": 0.0010000000474974513,
        "group": "Gauss-Newton options",
        "label": "Minimum model perturbation",
        "value": 0.0010000000474974513,
    },
    "reference_model": {
        "association": "Cell",
        "dataType": "Float",
        "default": 0.0010000000474974513,
        "group": "Model parameters",
        "isValue": True,
        "label": "Reference model",
        "main": False,
        "max": 100000000,
        "min": 9.99999993922529e-09,
        "parent": "mesh",
        "property": "",
        "value": 0.0010000000474974513,
    },
    "results_loaded": False,
    "smooth_mod": {
        "default": False,
        "group": "Model objective function",
        "label": "Reference model in Wxyz",
        "tooltip": "Wxyz(m-mref): SMOOTH_MOD_DIF option",
        "value": False,
    },
    "topography": {
        "association": "Cell",
        "dataType": ["Integer", "Boolean"],
        "default": "",
        "enabled": False,
        "label": "Topography model",
        "main": True,
        "optional": True,
        "parent": "mesh",
        "suffix": ".act",
        "value": "",
    },
    "update_ref": {
        "default": True,
        "group": "Model objective function",
        "label": "Update reference model",
        "value": True,
    },
    "upper_bound_model": {
        "association": "Cell",
        "dataType": "Float",
        "default": 10,
        "dependency": "bounds_defined",
        "dependencyType": "disabled",
        "group": "Model parameters",
        "isValue": True,
        "label": "Upper bounds",
        "main": False,
        "max": 100000000,
        "min": 9.99999993922529e-09,
        "parent": "mesh",
        "property": "",
        "value": 10,
    },
    "uuid": "",
    "version": "",
    "working_directory": "",
}
