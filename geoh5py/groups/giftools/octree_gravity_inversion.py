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
from geoh5py.groups.giftools.potential_field_base import POTENTIAL_FIELD_PARAMETERS


OCTGRVDEINVERSION_PARAMETERS: dict[str, Any] = OCTREE_INVERSION_PARAMETERS.copy()
OCTGRVDEINVERSION_PARAMETERS.pop(
    "global_weight", None
)  # this one isn't present in octgrvde
OCTGRVDEINVERSION_PARAMETERS.update(
    {
        "default_decay": {
            "default": False,
            "group": "Weight creation",
            "groupDependency": "cell_weight",
            "groupDependencyType": "disabled",
            "label": "Specify exponent decay",
            "value": False,
        },
        "depth_weighting_beta": merge_field(
            POTENTIAL_FIELD_PARAMETERS["depth_weighting_beta"],
            group="Weight creation",
            max=6.0,
            min=1e-3,
            value=2.0,
        ),
        "depth_weighting_z0": merge_field(
            POTENTIAL_FIELD_PARAMETERS["depth_weighting_z0"],
            group="Weight creation",
            max=1.0e6,
        ),
        "initial_model": merge_field(
            OCTREE_INVERSION_PARAMETERS["initial_model"],
            drop_keys=("max", "min"),
        ),
        "iterative_solver": merge_field(
            OCTREE_INVERSION_PARAMETERS["iterative_solver"],
            drop_keys="visible",
            value="Direct",
        ),
        "lower_bound_model": merge_field(
            OCTREE_INVERSION_PARAMETERS["lower_bound_model"],
            drop_keys=("max", "min"),
            default=-10.0,
            value=-10.0,
        ),
        "matlab": "OCTGRVDEinversion",
        "reference_model": merge_field(
            OCTREE_INVERSION_PARAMETERS["reference_model"],
            drop_keys=("max", "min"),
            default=0.0,
            value=0.0,
        ),
        "rx_data": {
            "default": "",
            "gifType": "GRAVdata",
            "label": "Data",
            "main": True,
            "meshType": "",
            "value": "",
        },
        "update_ref": merge_field(
            OCTREE_INVERSION_PARAMETERS["update_ref"],
            visible=False,
        ),
        "upper_bound_model": merge_field(
            OCTREE_INVERSION_PARAMETERS["upper_bound_model"],
            drop_keys=("max", "min"),
        ),
        "version": "1",
    }
)


class OCTGRVDEInversion(BaseGIFtoolsGroup):
    """Inversion group for UBC-OCTGRVDE."""

    _TYPE_UID = UUID("{4e043415-a0ea-4cef-bf89-2771e27b346c}")
    _default_name = "octgrvde"
    _default_parameters: dict[str, Any] = OCTGRVDEINVERSION_PARAMETERS
