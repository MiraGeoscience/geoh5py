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

import copy
from collections.abc import Iterable
from typing import Any

from geoh5py.groups.base import Group
from geoh5py.groups.giftools.giftools import GIFtoolsGroup, update_dict_parameters
from geoh5py.shared.utils import dict_mapper, entity2uuid, str2uuid


def merge_field(
    base_field: dict[str, Any],
    *,
    drop_keys: str | Iterable[str] = (),
    **overrides: Any,
) -> dict[str, Any]:
    """
    Return a deep copy of a shared parameter field with ``overrides`` applied.

    Can be used when a group shares a field with a common parameter dict (e.g.
    :data:`BASE_PARAMETERS` or similar) that needs a small difference, such as
    an extra key or a changed value. The deep copy ensures the per-group dict
    never aliases the shared template, so in-place edits cannot leak between
    groups.

    :param base_field: A shared field dict to use as the starting point.
    :param drop_keys: Key or keys to remove from the copied field before applying overrides.
    :param overrides: Keys to add or override on the copied field.
    :return: A new field dict, safe to embed in a group's parameter dict.
    """
    field = copy.deepcopy(base_field)
    if isinstance(drop_keys, str):
        drop_keys = (drop_keys,)
    for key in drop_keys:
        field.pop(key, None)
    field.update(overrides)
    return field


# Shared length-scale field reused across potential-field and octree inversion groups.
# Set ``is_length`` to True/False via merge_field when embedding in a parameter dict.
BASE_LENGTH_SCALES: dict[str, Any] = {
    "alpha_s": 1e-4,
    "alpha_x": 1.0,
    "alpha_y": 1.0,
    "alpha_z": 1.0,
    "group": "Model objective function",
    "length_x": 1.0e2,
    "length_y": 1.0e2,
    "length_z": 1.0e2,
    "parent": "mesh",
}

# Repository of shared field definitions reused across inversion groups.
# This dict is never spread whole into a group's parameter dict; individual
# fields are plucked by key or using merge_field when needed
BASE_PARAMETERS: dict[str, Any] = {
    "active_model": {
        "association": "Cell",
        "dataType": ["Integer", "Boolean"],
        "default": "",
        "enabled": False,
        "group": "Model parameters",
        "label": "Active model",
        "optional": True,
        "parent": "mesh",
        "suffix": ".act",
        "value": "",
    },
    "auto_threshold": {
        "choiceList": ["Relative error", "Specify value"],
        "default": "Relative error",
        "enabled": True,
        "group": "Wavelet compression",
        "groupOptional": True,
        "label": "Threshold mode",
        "value": "Relative error",
    },
    "bound_model_upper": {
        "association": "Cell",
        "dataType": "Float",
        "default": 10.0,
        "group": "Model parameters",
        "isValue": True,
        "label": "Bounds (upper)",
        "main": False,
        "parent": "mesh",
        "property": "",
        "value": 10.0,
    },
    "cell_weight": {
        "association": "Cell",
        "dataType": "Float",
        "default": "",
        "enabled": False,
        "group": "Model objective function",
        "label": "Weights (Ws)",
        "ndv": 1,
        "optional": True,
        "parent": "mesh",
        "suffix": ".wgt",
        "value": "",
    },
    "face_weight": {
        "association": "Face",
        "dataType": "Float",
        "default": "",
        "enabled": False,
        "group": "Model objective function",
        "label": "Weights (Wxyz)",
        "ndv": 1,
        "optional": True,
        "parent": "mesh",
        "suffix": ".wgt",
        "value": "",
    },
    "initial_model": {
        "association": "Cell",
        "dataType": "Float",
        "default": 1e-3,
        "group": "Model parameters",
        "isValue": True,
        "label": "Initial model",
        "main": False,
        "parent": "mesh",
        "property": "",
        "value": 1e-3,
    },
    "inversion_mode": {
        "choiceList": ["Chifactor", "Single beta"],
        "default": "Chifactor",
        "group": "Inversion stopping criteria",
        "label": "Inversion mode",
        "value": "Chifactor",
    },
    "inversion_par": {
        "default": 1.0,
        "group": "Inversion stopping criteria",
        "label": "Chifactor or beta",
        "min": 1e-12,
        "value": 1.0,
    },
    "inversion_tolerance": {
        "default": 5.0e-2,
        "group": "Inversion stopping criteria",
        "label": "Tolerance",
        "max": 0.5,
        "min": 1e-3,
        "value": 5.0e-2,
    },
    "length_scales": merge_field(BASE_LENGTH_SCALES, is_length=True),
    "matrix_file": {
        "enabled": False,
        "fileDescription": ["UBC matrix file"],
        "fileType": ["mtx"],
        "label": "Sensitivity file",
        "optional": True,
        "value": "",
    },
    "mesh": {
        "default": "",
        "label": "Mesh",
        "main": True,
        "meshType": "{b020a277-90e2-4cd7-84d6-612ee3f25051}",
        "value": "",
    },
    "reference_model": {
        "association": "Cell",
        "dataType": "Float",
        "default": 1.0e-8,
        "group": "Model parameters",
        "isValue": True,
        "label": "Reference model",
        "main": False,
        "parent": "mesh",
        "property": "",
        "value": 1.0e-8,
    },
    "results_loaded": False,
    "smooth_mod": {
        "default": False,
        "group": "Model objective function",
        "label": "Reference model in Wxyz",
        "tooltip": "Wxyz(m-mref): SMOOTH_MOD_DIF option",
        "value": False,
    },
    "threshold": {
        "default": 5.0e-2,
        "enabled": True,
        "group": "Wavelet compression",
        "groupOptional": True,
        "label": "Threshold value",
        "value": 5.0e-2,
    },
    "topography": {
        "default": "",
        "enabled": False,
        "label": "Topography",
        "main": True,
        "meshType": "{f26feba3-aded-494b-b9e9-b2bbcbe298e1}",
        "optional": True,
        "value": "",
    },
    "uuid": "",
    "wavelet": {
        "choiceList": [
            "daub1",
            "daub2",
            "daub3",
            "daub4",
            "daub5",
            "daub6",
            "symm4",
            "symm5",
            "symm6",
        ],
        "enabled": True,
        "group": "Wavelet compression",
        "groupOptional": True,
        "label": "Wavelet type",
        "value": "daub2",
    },
    "working_directory": "",
    "xy_localize": {
        "default": False,
        "label": "Localize coordinates",
        "main": True,
        "tooltip": "Writes files to disk with respect to UBC origin of 3D grid",
        "value": False,
    },
}


class BaseGIFtoolsGroup(Group):
    """
    Base class for the GIFtools application groups that live under a
    :obj:`~geoh5py.groups.giftools.giftools.GIFtoolsGroup` project and carry a
    ``parameters`` form.
    Concrete subclasses only need to define ``_TYPE_UID``, ``_default_name`` and
    ``_default_parameters``.
    """

    _default_parameters: dict[str, Any]

    def __init__(self, parameters: dict | None = None, **kwargs):
        self._parameters = self._validate_parameters(parameters)
        super().__init__(**kwargs)

    @property
    def parameters(self) -> dict:
        """
        Metadata attached to the entity.
        Return a copy of the dictionary to avoid accidental modifications.
        """
        return self._parameters.copy()

    def set_parameters(self, **kwargs):
        """
        Set parameter values in the 'parameters' dictionary.
        :param kwargs: Dictionary of parameters to modify.
        """
        if isinstance(self.parent, GIFtoolsGroup):
            self.parent.update_child_parameters(self.uid, **kwargs)
            self._parameters = {}  # Reset and rely on parent GIFtools group values
        else:
            update_dict_parameters(self._parameters, **kwargs)
        if self.on_file:
            self.workspace.update_attribute(self, "parameters")

    def _validate_parameters(self, value: dict | None) -> dict:
        """
        Validate and reformat the entries of the 'parameters' dictionary.
        :param value: Dictionary of parameters to modify.
        :return: Formatted dictionary of parameters.
        """
        if value is None:
            value = copy.deepcopy(self._default_parameters)
        if not isinstance(value, dict):
            raise TypeError(f"Input 'parameters' must be of type {dict}.")
        return dict_mapper(value, [str2uuid, entity2uuid])
