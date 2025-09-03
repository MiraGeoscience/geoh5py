# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2025 Mira Geoscience Ltd.                                     '
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

import json
from typing import Any
from uuid import UUID

import numpy as np

from geoh5py.groups import Group
from geoh5py.objects import ObjectBase
from geoh5py.shared.utils import (
    dict_mapper,
    entity2uuid,
    str2uuid,
    str_json_to_dict,
    stringify,
    uuid2entity,
)


class UIJsonGroup(Group):
    """
    Group for storing ui.json files.

    In Geoscience ANALYST, it can be used to store ui.jsons.

    :param options: Dictionary containing the ui.json structure.
    """

    _TYPE_UID = UUID("{BB50AC61-A657-4926-9C82-067658E246A0}")
    _default_name = "UIJson"

    def __init__(
        self,
        options: dict | np.ndarray | bytes | None = None,
        **kwargs,
    ):
        self._options: dict

        super().__init__(**kwargs)

        self.options = self.format_input_options(options)

    def _copy_relatives(self, parent, clear_cache: bool = False):
        """
        Copy the relatives of the entity.

        :param parent: The parent to copy the entity to.
        :param clear_cache: Indicate whether to clear the cache.
        """

        if (
            parent is None
            or parent == self.parent
            or parent == self.workspace
            or len(self.options) == 0
        ):
            return

        def copy_obj_and_group(val: Any) -> Any:
            """
            Function to copy objects and groups found in the options.
            To be used in dict_mapper for intricate structures.

            :param val: The value to check and possibly copy.

            :return: The same value
            """
            val = uuid2entity(str2uuid(val), self.workspace)
            if isinstance(val, Group | ObjectBase):
                val.copy(parent, copy_children=True, clear_cache=clear_cache)

            return val

        dict_mapper(self.options, [copy_obj_and_group])

    def copy(
        self,
        parent=None,
        *,
        copy_children: bool = False,
        copy_relatives: bool = True,
        clear_cache: bool = False,
        **kwargs,
    ) -> UIJsonGroup | None:
        """
        Sub-class extension of :func:`~geoh5py.groups.base.Group.copy`.

        :param parent: The parent to copy the entity to.
        :param copy_children: Whether to copy the children of the entity.
        :param copy_relatives: If true, the objects and groups referenced in the options are copied.
        :param clear_cache: Indicate whether to clear the cache.
        :param kwargs: other keyword arguments.

        :return: The copied entity.
        """
        if copy_relatives:
            self._copy_relatives(parent, clear_cache=clear_cache)

        copied = super().copy(
            parent=parent,
            copy_children=copy_children,
            clear_cache=clear_cache,
            **kwargs,
        )

        return copied

    @property
    def options(self) -> dict:
        """
        Metadata attached to the entity.

        Return a copy of the dictionary to avoid accidental modifications.
        """
        return self._options.copy()

    @options.setter
    def options(self, value: dict):
        if not isinstance(value, dict):
            raise TypeError(f"Input 'options' must be of type {dict}.")

        self._options = dict_mapper(value, [str2uuid, entity2uuid])
        if self.on_file:
            self.workspace.update_attribute(self, "options")

    def add_ui_json(self, name: str | None = None):
        """
        Add ui.json file to entity.

        :param name: Name of the file in the workspace.
        """
        if not self._options:
            raise ValueError("UIJsonGroup must have options set.")

        json_str = json.dumps(stringify(self.options, [entity2uuid]), indent=4)
        bytes_data = json_str.encode("utf-8")

        if name is None:
            name = self.name

        self.add_file(bytes_data, name=f"{name}.ui.json")

    @staticmethod
    def format_input_options(options: dict | np.ndarray | bytes | None) -> dict:
        """
        Format input options to a dictionary.

        :param options: Input options.

        :return: Formatted options.
        """
        if options is None:
            return {}

        if isinstance(options, np.ndarray):
            options = options[0]

        if isinstance(options, bytes):
            options = str_json_to_dict(options)

        return options
