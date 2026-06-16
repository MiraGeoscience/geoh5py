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

from geoh5py.groups.base import Group
from geoh5py.groups.giftools.giftools import GIFtoolsGroup, update_dict_parameters
from geoh5py.shared.utils import dict_mapper, entity2uuid, str2uuid


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
            value = self._default_parameters.copy()
        if not isinstance(value, dict):
            raise TypeError(f"Input 'parameters' must be of type {dict}.")
        return dict_mapper(value, [str2uuid, entity2uuid])
