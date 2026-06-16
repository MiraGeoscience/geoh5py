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
from geoh5py.groups.giftools.octree_base import OCTREE_INVERSION_PARAMETERS


DCOCTREE_PARAMETERS = OCTREE_INVERSION_PARAMETERS.copy()
DCOCTREE_PARAMETERS.update(
    {
        "matlab": "DCoctreeinversion",
        "rx_data": {
            "default": "",
            "gifType": "DC3Ddata",
            "label": "Data",
            "main": True,
            "meshType": "",
            "value": "",
        },
        "xy_localize": {
            "default": False,
            "label": "Localize coordinates",
            "main": True,
            "tooltip": "Writes files to disk with respect to UBC origin of 3D grid",
            "value": False,
        },
        "version": "20200508",
    }
)


class DCOctreeInversion(BaseGIFtoolsGroup):
    """Inversion group for UBC-DCOctree."""

    _TYPE_UID = UUID("{54d296de-0588-472c-9a62-480098303394}")
    _default_name = "dcoctree_inv"
    _default_parameters: dict[str, Any] = DCOCTREE_PARAMETERS
