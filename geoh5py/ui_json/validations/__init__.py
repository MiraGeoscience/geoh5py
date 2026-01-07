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

from collections.abc import Callable

from .uijson import (
    ErrorPool,
    UIJsonError,
    dependency_type_validation,
    mesh_type_validation,
    parent_validation,
)


VALIDATIONS_MAP = {
    "dependency": dependency_type_validation,
    "mesh_type": mesh_type_validation,
    "parent": parent_validation,
}


def get_validations(form_keys: list[str]) -> list[Callable]:
    """Returns a list of callable validations based on identifying form keys."""
    return [VALIDATIONS_MAP[k] for k in form_keys if k in VALIDATIONS_MAP]
