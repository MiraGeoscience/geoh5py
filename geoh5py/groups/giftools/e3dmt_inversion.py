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

from geoh5py.groups.giftools.base import BaseGIFtoolsGroup, merge_field
from geoh5py.groups.giftools.inversion_base import (
    ASSIGN_CON_RES_FIELD,
    SUSCEPTIBILITY_FIELD,
)
from geoh5py.groups.giftools.octree_base import OCTREE_INVERSION_PARAMETERS


# Fields unique to e3dmt. Only a subset of BASE_PARAMETERS is used

E3DMTINV_PARAMETERS: dict[str, Any] = OCTREE_INVERSION_PARAMETERS.copy()
E3DMTINV_PARAMETERS.update(
    {
        "ModeConRes": merge_field(
            ASSIGN_CON_RES_FIELD,
            drop_keys="group",
            main=True,
        ),
        "bicg_ipcg_tolerance": {
            "default": 1e-5,
            "group": "Forward solver (BiCGstab) options",
            "label": "IPCG tolerance",
            "max": 1e-1,
            "min": 1e-14,
            "value": 1e-5,
        },
        "bicg_iterations": {
            "default": 15,
            "dependency": "pardiso",
            "dependencyType": "disabled",
            "enabled": True,
            "group": "Forward solver (BiCGstab) options",
            "label": "Maximum iterations",
            "max": 500,
            "min": 1,
            "value": 150,
            "visible": True,
        },
        "bicg_tolerance": {
            "default": 1e-11,
            "dependency": "pardiso",
            "dependencyType": "disabled",
            "enabled": True,
            "group": "Forward solver (BiCGstab) options",
            "label": "Forward tolerance",
            "max": 1e-1,
            "min": 1e-20,
            "value": 1e-11,
            "visible": True,
        },
        "initial_model": merge_field(
            OCTREE_INVERSION_PARAMETERS["initial_model"],
            default=1e-2,
            value=1e-2,
        ),
        "inversion_chifact": merge_field(
            OCTREE_INVERSION_PARAMETERS["inversion_chifact"],
            value=5e-1,
        ),
        "ipcg_tolerance": merge_field(
            OCTREE_INVERSION_PARAMETERS["ipcg_tolerance"],
            dependency="pardiso",
            dependencyType="disabled",
            enabled=True,
            visible=True,
        ),
        "iterative_solver": merge_field(
            OCTREE_INVERSION_PARAMETERS["iterative_solver"],
            group="Forward solver (BiCGstab) options",
            visible=True,
        ),
        "iwt": {
            "label": "Positive iwt",
            "main": True,
            "tooltip": "Uses positive e^iwt if checked, otherwise e^-iwt",
            "value": False,
            "visible": False,
        },
        "matlab": "E3DMTinversion",
        "minimum_frequency": {
            "default": -1e0,
            "enabled": False,
            "group": "Forward solver (BiCGstab) options",
            "label": "Minimum SSOR frequency (Hz)",
            "optional": True,
            "value": -1e0,
            "visible": False,
        },
        "model_1d_con": {
            "default": 1e-2,
            "label": "1D conductivity",
            "main": True,
            "value": 1e-2,
        },
        "model_perturbation": merge_field(
            OCTREE_INVERSION_PARAMETERS["model_perturbation"],
            visible=True,
        ),
        "nbeta": {
            "group": "Gauss-Newton options",
            "label": "Maximum number of Beta iterations",
            "max": 1000,
            "min": 1,
            "value": 10,
            "visible": False,
        },
        "reference_model": merge_field(
            OCTREE_INVERSION_PARAMETERS["reference_model"],
            default=1e-2,
            value=1e-2,
        ),
        "rx_data": {
            "default": "",
            "gifType": ["IMPdata", "ZTEMdata"],
            "label": "Data",
            "main": True,
            "meshType": "",
            "value": "",
        },
        "susceptibility": merge_field(
            SUSCEPTIBILITY_FIELD,
            drop_keys="tooltip",
            label="Background susceptibility model",
        ),
        "upper_bound_model": merge_field(
            OCTREE_INVERSION_PARAMETERS["upper_bound_model"],
            default=1e2,
            value=1e2,
        ),
        "version": "1",
        "write_to_disk": {
            "group": "Gauss-Newton options",
            "label": "Write factorizations to disk",
            "tooltip": "Slower but capable of solving much larger problems",
            "value": False,
            "visible": False,
        },
        "ztem_data": {
            "default": "",
            "enabled": False,
            "group": "Joint inversion",
            "groupOptional": True,
            "label": "Tipper Data",
            "meshType": "ZTEMdata",
            "value": "",
            "visible": True,
        },
        "ztem_data_wght": {
            "enabled": False,
            "group": "Joint inversion",
            "groupOptional": True,
            "label": "Data weighting constant",
            "max": 9.95e-1,
            "min": 5e-3,
            "precision": 3,
            "value": 5e-1,
            "visible": True,
        },
    }
)


class E3DMTInv(BaseGIFtoolsGroup):
    """Inversion group for UBC-E3DMT."""

    _TYPE_UID = UUID("{8cf239e3-63a6-4813-adf8-9714293b602e}")
    _default_name = "e3dmt_iter"
    _default_parameters: dict[str, Any] = E3DMTINV_PARAMETERS
