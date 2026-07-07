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
    BASE_LENGTH_SCALES,
    BASE_PARAMETERS,
    merge_field,
)


# Parameters shared by the GIFtools octree inversion groups.
# Note -- this will likely be restructured as we know more about what the commonalities and
# differences are between different GIFtools/GIFtools octree/GIFtools inversion groups.

OCTREE_INVERSION_PARAMETERS: dict[str, Any] = {
    "active_model": BASE_PARAMETERS["active_model"],
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
        "value": 1e-3,
    },
    "beta_two": {
        "default": -1234567,
        "dependency": "beta_given",
        "group": "Inversion parameters",
        "label": "Beta maximum",
        "min": 1e-10,
        "value": 1.0e3,
    },
    "bounds_defined": {
        "group": "Model parameters",
        "label": "No bounds",
        "value": True,
    },
    "cell_weight": BASE_PARAMETERS["cell_weight"],
    "delta_beta": {
        "default": -1234567,
        "dependency": "beta_given",
        "group": "Inversion parameters",
        "label": "Beta step",
        "value": 0.25,
    },
    "face_weight": BASE_PARAMETERS["face_weight"],
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
        "default": 1e-2,
        "group": "Gauss-Newton options",
        "label": "Solver tolerance",
        "value": 1e-2,
    },
    # initial_model shares the common base and octree adds max/min bounds.
    "initial_model": merge_field(
        BASE_PARAMETERS["initial_model"],
        max=1.0e8,
        min=1.0e-8,
    ),
    "inversion_chifact": {
        "default": 1.0,
        "group": "Inversion parameters",
        "label": "Chi factor",
        "value": 1.0,
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
        "default": 1e-2,
        "group": "Gauss-Newton options",
        "label": "IPCG tolerance",
        "tooltip": "Fractional percent norm of the iterative solver residual",
        "value": 1e-2,
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
    "length_scales": merge_field(BASE_LENGTH_SCALES, is_length=False),
    "lower_bound_model": {
        "association": "Cell",
        "dataType": "Float",
        "default": 1.0e-8,
        "dependency": "bounds_defined",
        "dependencyType": "disabled",
        "group": "Model parameters",
        "isValue": True,
        "label": "Lower bounds",
        "main": False,
        "max": 1.0e8,
        "min": 1.0e-8,
        "parent": "mesh",
        "property": "",
        "value": 1.0e-8,
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
        "default": 1e-3,
        "group": "Gauss-Newton options",
        "label": "Minimum model perturbation",
        "value": 1e-3,
    },
    "reference_model": {
        "association": "Cell",
        "dataType": "Float",
        "default": 1e-3,
        "group": "Model parameters",
        "isValue": True,
        "label": "Reference model",
        "main": False,
        "max": 1.0e8,
        "min": 1.0e-8,
        "parent": "mesh",
        "property": "",
        "value": 1e-3,
    },
    "results_loaded": BASE_PARAMETERS["results_loaded"],
    "smooth_mod": BASE_PARAMETERS["smooth_mod"],
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
        "default": 10.0,
        "dependency": "bounds_defined",
        "dependencyType": "disabled",
        "group": "Model parameters",
        "isValue": True,
        "label": "Upper bounds",
        "main": False,
        "max": 1.0e8,
        "min": 1.0e-8,
        "parent": "mesh",
        "property": "",
        "value": 10.0,
    },
    "uuid": BASE_PARAMETERS["uuid"],
    "version": "",
    "working_directory": BASE_PARAMETERS["working_directory"],
    "xy_localize": {
        "default": False,
        "label": "Localize coordinates",
        "main": True,
        "tooltip": "Writes files to disk with respect to UBC origin of 3D grid",
        "value": False,
    },
}
