#  Copyright (c) 2024 Mira Geoscience Ltd.
#
#  This file is part of geoh5py.
#
#  geoh5py is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  geoh5py is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with geoh5py.  If not, see <https://www.gnu.org/licenses/>.
from __future__ import annotations

import json
import uuid

import numpy as np

from ..shared.utils import str_json_to_dict, stringify
from .base import Group


class UIJsonGroup(Group):
    """Group for SimPEG inversions."""

    _TYPE_UID = uuid.UUID("{BB50AC61-A657-4926-9C82-067658E246A0}")
    _default_name = "UIJson"

    def __init__(
        self,
        options: dict | np.ndarray | bytes | None = None,
        **kwargs,
    ):
        self._options: dict

        super().__init__(**kwargs)
        self.options = self.format_input_options(options)

    @property
    def options(self) -> dict:
        """
        Metadata attached to the entity.
        """
        return self._options

    @options.setter
    def options(self, value: dict):
        if not isinstance(value, dict):
            raise ValueError(f"Input 'options' must be of type {dict}.")

        self._options = value

        if self.on_file:
            self.workspace.update_attribute(self, "options")

    def add_ui_json(self, name: str | None = None):
        """
        Add ui.json file to entity.

        :param name: Name of the file in the workspace.
        """
        if self.options is None:
            raise ValueError("UIJsonGroup must have options set.")

        json_str = json.dumps(stringify(self.options), indent=4)
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

        if not isinstance(options, dict):
            raise ValueError(f"Input 'options' must be of type {dict}.")

        return options
