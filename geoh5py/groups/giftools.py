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

from geoh5py.shared.utils import (
    dict_mapper,
    entity2uuid,
    str2uuid,
    stringify,
)

from .base import Group


class GIFtoolsGroup(Group):
    """The type for a GIFtools group."""

    _TYPE_UID = UUID(fields=(0x585B3218, 0xC24B, 0x41FE, 0xAD, 0x1F, 0x24D5E6E8348A))
    _default_name = "GIFtools Project"

    def __init__(self, gif_parameters: list[dict] | None = None, **kwargs):

        super().__init__(**kwargs)
        self._gif_parameters = self._validate_parameters(gif_parameters)

    @property
    def gif_parameters(self) -> list[dict]:
        """
        Metadata attached to the entity.

        Return a copy of the dictionary to avoid accidental modifications.
        """
        return self._gif_parameters.copy()

    @staticmethod
    def _validate_parameters(value: list[dict] | None) -> list[dict]:

        if value is None:
            value = []

        if not isinstance(value, list):
            raise TypeError(f"Input 'gif_parameters' must be of type {dict}.")

        return dict_mapper(value, [str2uuid, entity2uuid])


class GIFtoolInversion(Group):
    """Base class for all GIFtools inversion groups."""

    def __init__(self, parameters: dict | None = None, **kwargs):

        super().__init__(**kwargs)
        self._parameters = self._validate_parameters(parameters)

    @property
    def parameters(self) -> dict:
        """
        Metadata attached to the entity.

        Return a copy of the dictionary to avoid accidental modifications.
        """
        return self._parameters.copy()

    def modify_parameters(self, **kwargs):
        """
        Modify a single parameter in the 'parameters' dictionary.

        :param kwargs: Dictionary of parameters to modify.
        """
        for key, value in kwargs.items():
            if key not in INV_PARAMETERS:
                raise ValueError(f"Parameter '{key}' is not supported.")

            if "value" in INV_PARAMETERS[key]:
                self._parameters[key]["value"] = stringify(value)
            else:
                self._parameters[key] = stringify(value)

        if self.on_file:
            self.workspace.update_attribute(self, "parameters")

    @staticmethod
    def _validate_parameters(value: dict | None) -> dict:
        """
        Validate and reformat the entries of the 'parameters' dictionary.

        :param value: Dictionary of parameters to modify.
        :return: Formatted dictionary of parameters.
        """
        if value is None:
            value = INV_PARAMETERS

        if not isinstance(value, dict):
            raise TypeError(f"Input 'parameters' must be of type {dict}.")

        return dict_mapper(value, [str2uuid, entity2uuid])


class DCOctreeInversion(GIFtoolInversion):
    """Inversion group for UBC-DCOctree."""

    _TYPE_UID = UUID("{54d296de-0588-472c-9a62-480098303394}")
    _default_name = "dcoctree_inv"


INV_PARAMETERS: dict[str, Any] = {
    "active_model": {"enabled": False, "value": ""},
    "assignConRes": {"enabled": True, "value": "Conductivity"},
    "beta_given": {"enabled": True, "value": False},
    "beta_one": {"enabled": False, "placeholderText": "0.001", "value": 0.001},
    "beta_two": {"enabled": False, "placeholderText": "1000", "value": 1000},
    "bounds_defined": {"enabled": True, "value": True},
    "cell_weight": {"enabled": False, "value": ""},
    "delta_beta": {"enabled": False, "placeholderText": "0.25", "value": 0.25},
    "face_weight": {"enabled": False, "value": ""},
    "global_weight": {"enabled": False, "value": ""},
    "gn_tolerance": {"enabled": True, "placeholderText": "0.01", "value": 0.01},
    "initial_model": {
        "enabled": True,
        "isValue": True,
        "placeholderText": "0.001",
        "value": 0.0010000000474974513,
    },
    "inversion_chifact": {"enabled": True, "placeholderText": "1", "value": 1},
    "ipcg_iterations": {"enabled": True, "value": 20},
    "ipcg_tolerance": {"enabled": True, "placeholderText": "0.01", "value": 0.01},
    "iterations_per_beta": {"enabled": True, "value": 3},
    "iterative_solver": {"enabled": True, "value": "Iterative"},
    "length_scales": {
        "alpha_s": 1e-4,
        "alpha_x": 1,
        "alpha_y": 1,
        "alpha_z": 1,
        "enabled": True,
        "is_length": False,
        "length_x": 100,
        "length_y": 100,
        "length_z": 100,
    },
    "lower_bound_model": {
        "enabled": False,
        "isValue": True,
        "placeholderText": "0",
        "value": 0,
    },
    "mesh": {"enabled": True, "value": ""},
    "model_perturbation": {"enabled": True, "placeholderText": "0.001", "value": 0.001},
    "reference_model": {
        "enabled": True,
        "isValue": True,
        "placeholderText": "0.001",
        "value": 0.0010000000474974513,
    },
    "rx_data": {"enabled": True, "value": ""},
    "smooth_mod": {"enabled": True, "value": False},
    "topography": {"enabled": False, "value": ""},
    "update_ref": {"enabled": True, "value": True},
    "upper_bound_model": {
        "enabled": False,
        "isValue": True,
        "placeholderText": "10",
        "value": 10,
    },
    "xy_localize": {"enabled": True, "value": False},
}
