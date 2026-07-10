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


ASSIGN_CON_RES_FIELD: dict[str, Any] = {
    "alternateLabel": "Resistivity",
    "group": "Model parameters",
    "label": "Model Type",
    "main": False,
    "originalLabel": "Conductivity",
    "tooltip": "Resistivity (Ohm-m) or Conductivity (S/m)",
    "value": "Conductivity",
}

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

# Generic GIFtools inversion controls reused by several UBC inversion groups.
INVERSION_CONTROL_PARAMETERS: dict[str, Any] = {
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
    "delta_beta": {
        "default": -1234567,
        "dependency": "beta_given",
        "group": "Inversion parameters",
        "label": "Beta step",
        "value": 2.5e-1,
    },
    "gn_tolerance": {
        "default": 1e-2,
        "group": "Gauss-Newton options",
        "label": "Solver tolerance",
        "value": 1e-2,
    },
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
    "model_perturbation": {
        "default": 1e-3,
        "group": "Gauss-Newton options",
        "label": "Minimum model perturbation",
        "value": 1e-3,
    },
    "update_ref": {
        "default": True,
        "group": "Model objective function",
        "label": "Update reference model",
        "value": True,
    },
}
