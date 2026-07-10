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
from geoh5py.groups.giftools.inversion_base import ASSIGN_CON_RES_FIELD
from geoh5py.groups.giftools.octree_base import OCTREE_INVERSION_PARAMETERS


# Fields unique to e3d. Only a subset of BASE_PARAMETERS is used

E3DINV_PARAMETERS: dict[str, Any] = OCTREE_INVERSION_PARAMETERS.copy()
E3DINV_PARAMETERS.update(
    {
        "assignConRes": ASSIGN_CON_RES_FIELD,
        "initial_model": merge_field(
            OCTREE_INVERSION_PARAMETERS["initial_model"],
            default=1e-3,
            value=1e-3,
        ),
        "lower_bound_model": merge_field(
            OCTREE_INVERSION_PARAMETERS["lower_bound_model"],
            default=1e-8,
            value=1e-8,
        ),
        "matlab": "E3Dinversion",
        "mesh": merge_field(
            OCTREE_INVERSION_PARAMETERS["mesh"],
            group="Mesh and models",
        ),
        "reference_model": merge_field(
            OCTREE_INVERSION_PARAMETERS["reference_model"],
            default=1e-3,
            value=1e-3,
        ),
        "rx_data": {
            "default": "",
            "gifType": ["FEMdata"],
            "group": "Data options",
            "label": "Data",
            "main": True,
            "meshType": "",
            "value": "",
        },
        "susceptibility": {
            "association": "Cell",
            "dataType": "Float",
            "default": 1e-8,
            "enabled": False,
            "group": "Model parameters",
            "isValue": True,
            "label": "Susceptibility (SI)",
            "main": False,
            "optional": True,
            "parent": "mesh",
            "property": "",
            "tooltip": "Susceptibility (SI)",
            "value": 1e-8,
        },
        "topography": merge_field(
            OCTREE_INVERSION_PARAMETERS["topography"],
            association="Cell",
            dataType=["Integer", "Boolean"],
            group="Mesh and models",
            label="Topography model",
            main=True,
            optional=True,
            parent="mesh",
            suffix=".act",
        ),
        "upper_bound_model": merge_field(
            OCTREE_INVERSION_PARAMETERS["upper_bound_model"],
            default=1e3,
            value=1e3,
        ),
        "version": "1",
    }
)


class E3DInv(BaseGIFtoolsGroup):
    """Inversion group for UBC-E3D."""

    _TYPE_UID = UUID("{9a0b9d39-9e6d-409e-a7cd-ffc72474feed}")
    _default_name = "e3d"
    _default_parameters: dict[str, Any] = E3DINV_PARAMETERS
