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
from geoh5py.groups.giftools.octree_base import OCTREE_INVERSION_PARAMETERS


IPOCTREEINV_PARAMETERS = OCTREE_INVERSION_PARAMETERS.copy()
IPOCTREEINV_PARAMETERS.update(
    {
        "matlab": "IPoctreeinversion",
        "modeConRes": {
            "alternateLabel": "Resistivity",
            "label": "Model Type",
            "main": True,
            "originalLabel": "Conductivity",
            "tooltip": "Resistivity (Ohm-m) or Conductivity (S/m)",
            "value": "Conductivity",
        },
        "model_conductivity": {
            "association": "Cell",
            "dataType": "Float",
            "label": "Conductivity model",
            "main": True,
            "ndv": 1.0e-8,
            "parent": "mesh",
            "suffix": ".con",
            "value": "",
        },
        "reference_model": merge_field(
            OCTREE_INVERSION_PARAMETERS["reference_model"], default=1.0e-8, value=1.0e-8
        ),
        "rx_data": {
            "default": "",
            "gifType": "IP3Ddata",
            "label": "Data",
            "main": True,
            "meshType": "",
            "value": "",
        },
        "version": "20200508",
    }
)


class IPOctreeInversion(BaseGIFtoolsGroup):
    """Inversion group for UBC-IPOctree."""

    _TYPE_UID = UUID("{d9fd455e-ea94-40f5-9d86-e7c49c7b5005}")
    _default_name = "ipoctree_inv"
    _default_parameters: dict[str, Any] = IPOCTREEINV_PARAMETERS
