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

from geoh5py.groups.giftools.base import BaseGIFtoolsGroup
from geoh5py.groups.giftools.parameters import BASE_PARAMETERS, merge_field
from geoh5py.groups.giftools.potential_field_base import (
    BOUND_MODEL_LOWER_FIELD,
    POTENTIAL_FIELD_PARAMETERS,
)


# Fields unique to mviinv, in addition to the shared potential field parameters.

MVIINV_PARAMETERS: dict[str, Any] = {
    **POTENTIAL_FIELD_PARAMETERS,
    "bound_model_lower": merge_field(
        BOUND_MODEL_LOWER_FIELD,
        drop_keys="dataType",
        dataGroupType="3D vector",
    ),
    "bound_model_upper": merge_field(
        POTENTIAL_FIELD_PARAMETERS["bound_model_upper"],
        drop_keys="dataType",
        dataGroupType="3D vector",
    ),
    "cell_weight": merge_field(
        POTENTIAL_FIELD_PARAMETERS["cell_weight"],
        label="Inducing cell weights (Ws)",
    ),
    "data": {
        "default": "",
        "gifType": "MAGdata",
        "label": "Data",
        "main": True,
        "meshType": "",
        "value": "",
    },
    "default_decay": merge_field(
        POTENTIAL_FIELD_PARAMETERS["default_decay"],
        visible=False,
    ),
    "depth_weighting": merge_field(
        POTENTIAL_FIELD_PARAMETERS["depth_weighting"],
        visible=False,
    ),
    "depth_weighting_beta": merge_field(
        POTENTIAL_FIELD_PARAMETERS["depth_weighting_beta"],
        visible=False,
    ),
    "depth_weighting_z0": merge_field(
        POTENTIAL_FIELD_PARAMETERS["depth_weighting_z0"],
        visible=False,
    ),
    "gamma": {
        "default": 1,
        "group": "Model parameters",
        "label": "Remanent/induced trade-off",
        "max": 100000,
        "min": 0,
        "value": 1,
    },
    "initial_model": merge_field(
        BASE_PARAMETERS["initial_model"],
        drop_keys="dataType",
        dataGroupType="3D vector",
    ),
    "matlab": "MVIinversion",
    "phi_qx": {
        "enabled": False,
        "group": "Angle Lq scaling values",
        "groupOptional": True,
        "label": "Phi Easting",
        "value": 0.009999999776482582,
    },
    "phi_qy": {
        "enabled": False,
        "group": "Angle Lq scaling values",
        "groupOptional": True,
        "label": "Phi Northing",
        "value": 0.009999999776482582,
    },
    "phi_qz": {
        "enabled": False,
        "group": "Angle Lq scaling values",
        "groupOptional": True,
        "label": "Phi Vertical",
        "value": 0.009999999776482582,
    },
    "reference_model": merge_field(
        POTENTIAL_FIELD_PARAMETERS["reference_model"],
        drop_keys="dataType",
        dataGroupType="3D vector",
    ),
    "remanent_weighting": {
        "association": "Cell",
        "dataType": "Float",
        "default": "",
        "enabled": False,
        "group": "Model objective function",
        "label": "Remanent cell weights (Ws)",
        "optional": True,
        "parent": "mesh",
        "value": "",
    },
    "theta_qx": {
        "enabled": False,
        "group": "Angle Lq scaling values",
        "groupOptional": True,
        "label": "Theta Easting",
        "value": 0.009999999776482582,
    },
    "theta_qy": {
        "enabled": False,
        "group": "Angle Lq scaling values",
        "groupOptional": True,
        "label": "Theta Northing",
        "value": 0.009999999776482582,
    },
    "theta_qz": {
        "enabled": False,
        "group": "Angle Lq scaling values",
        "groupOptional": True,
        "label": "Theta Vertical",
        "value": 0.009999999776482582,
    },
    "uuid": BASE_PARAMETERS["uuid"],
    "version": "3",
    "wavelet_diagnostics": merge_field(
        POTENTIAL_FIELD_PARAMETERS["wavelet_diagnostics"],
        visible=False,
    ),
}


class MVIInv(BaseGIFtoolsGroup):
    """Inversion group for UBC-MVIInv (mviinv)."""

    _TYPE_UID = UUID("{9472b5cb-a285-4257-a2e8-68a3d33aa1f2}")
    _default_name = "mviinv"
    _default_parameters: dict[str, Any] = MVIINV_PARAMETERS
