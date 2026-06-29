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
import copy
from collections.abc import Iterable
from typing import Any


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
        "default": 0.0010000000474974513,
        "group": "Model parameters",
        "isValue": True,
        "label": "Initial model",
        "main": False,
        "parent": "mesh",
        "property": "",
        "value": 0.0010000000474974513,
    },
    "results_loaded": False,
    "smooth_mod": {
        "default": False,
        "group": "Model objective function",
        "label": "Reference model in Wxyz",
        "tooltip": "Wxyz(m-mref): SMOOTH_MOD_DIF option",
        "value": False,
    },
    "uuid": "",
    "working_directory": "",
}
