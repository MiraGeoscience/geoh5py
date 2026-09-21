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

from geoh5py.groups.giftools.base import BASE_PARAMETERS, BaseGIFtoolsGroup
from geoh5py.groups.giftools.inversion_base import ASSIGN_CON_RES_FIELD
from geoh5py.groups.giftools.octree_base import OCTREE_INVERSION_PARAMETERS


DCOCTREE_PARAMETERS = OCTREE_INVERSION_PARAMETERS.copy()
DCOCTREE_PARAMETERS.update(
    {
        "assignConRes": ASSIGN_CON_RES_FIELD,
        "matlab": "DCoctreeinversion",
        "rx_data": {
            "default": "",
            "gifType": "DC3Ddata",
            "label": "Data",
            "main": True,
            "meshType": "",
            "value": "",
        },
        "version": "20200508",
    }
)


class DCOctreeInversion(BaseGIFtoolsGroup):
    """Inversion group for UBC-DCOctree."""

    _TYPE_UID = UUID("{54d296de-0588-472c-9a62-480098303394}")
    _default_name = "dcoctree_inv"
    _default_parameters: dict[str, Any] = DCOCTREE_PARAMETERS


DCOCTREE_FORWARD_PARAMETERS = BASE_PARAMETERS.copy()
DCOCTREE_FORWARD_PARAMETERS.update(
    {
        "mesh": {
            "default": "",
            "label": "Octree mesh",
            "main": True,
            "meshType": "{4ea87376-3ece-438b-bf12-3479733ded46}",
            "value": "",
        },
        "IPL": {
            "group": "Induced polarization",
            "groupDependency": "model_ip",
            "label": "Use linearized sensitivity matrix (IPL)",
            "main": False,
            "value": False,
        },
        "data": {
            "default": "",
            "gifType": ["DC3Ddata", "IP3Ddata"],
            "label": "Data",
            "main": True,
            "meshType": "",
            "value": "{00000000-0000-0000-0000-000000000000}",
        },
        "ip_type": {
            "choiceList": ["Apparent chargeability", "Secondary potential"],
            "default": "Apparent chargeability",
            "group": "Induced polarization",
            "groupDependency": "model_ip",
            "label": "Type of data",
            "main": False,
            "value": "Apparent chargeability",
        },
        "matlab": "dcipoctree_fwd",
        "model": {
            "association": "Cell",
            "dataType": "Float",
            "default": "",
            "label": "Conductivity model",
            "main": True,
            "ndv": 9.99999993922529e-09,
            "parent": "mesh",
            "value": "",
        },
        "model_ip": {
            "association": "Cell",
            "dataType": "Float",
            "default": "",
            "enabled": False,
            "label": "Chargeability model",
            "main": True,
            "ndv": 0,
            "optional": True,
            "parent": "mesh",
            "suffix": ".chg",
            "value": "",
        },
        "version": "20200508",
    }
)


class DCOctreeForward(BaseGIFtoolsGroup):
    """Forward group for UBC-DCOctree."""

    _TYPE_UID = UUID("{a522d641-6cb7-421b-836b-a14c0d9c7801}")
    _default_name = "dcoctree_fwd"
    _default_parameters: dict[str, Any] = DCOCTREE_FORWARD_PARAMETERS
