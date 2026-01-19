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

from typing import TYPE_CHECKING, Any

import numpy as np


if TYPE_CHECKING:
    from geoh5py.ui_json.forms import BaseForm


def all_subclasses(type_object: type[BaseForm]) -> list[type[BaseForm]]:
    """Recursively find all subclasses of input type object."""
    collection = []
    subclasses = type_object.__subclasses__()
    collection += subclasses
    for subclass in subclasses:
        collection += all_subclasses(subclass)
    return collection


def model_fields_difference(parent: type[BaseForm], child: type[BaseForm]) -> set[str]:
    """Isolate fields added in the child class."""
    return set(child.model_fields) - set(parent.model_fields)


def indicator_attributes(
    parent: type[BaseForm], children: list[type[BaseForm]]
) -> list[set[str]]:
    """List all the attributes defined in a subclass."""
    return [model_fields_difference(parent, c) for c in children]


def count_indicators(indicators: list[set[str]], data: dict[str, Any]) -> np.ndarray:
    """Return the number of matching indicators for each child class."""
    return np.array([len(i.intersection(data)) for i in indicators])


def filter_candidates_by_indicator_polling(
    indicators: list[set[str]],
    data: dict[str, Any],
    candidates: list[type[BaseForm]],
) -> list[type[BaseForm]]:
    """Return candidate subclass(es) with most matching indicators."""
    n_indicators = count_indicators(indicators, data)
    candidates = np.array(candidates)[n_indicators == np.max(n_indicators)]

    candidates = baseclass_if_overlapping(candidates, len(indicators))

    return candidates


def baseclass_if_overlapping(
    candidates: list[type[BaseForm]],
    n_children: int,
):
    if len(candidates) > 1 and len(candidates) < n_children:
        n_attributes = [len(k.model_fields) for k in candidates]
        candidates = candidates[n_attributes == np.min(n_attributes)]

    return candidates
