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

import logging
from pathlib import Path
from typing import Any

from geoh5py.ui_json import UIJson

DEFAULT_UI_JSON_NAME = "quick_run.ui.json"
logger = logging.getLogger(__name__)


class InputFile:
    """
    Handles loading ui.json input files.

    Attributes
    ----------
    data : Input file content parsed to flat dictionary of key:value.
    ui_json: User interface serializable as ui.json format
    workspace: Target :obj:`geoh5py.workspace.Workspace`
    validations: Dictionary of validations for parameters in the input file

    Methods
    -------
    write_ui_json()
        Writes a ui.json formatted file from 'data' attribute contents.
    read_ui_json()
        Reads a ui.json formatted file into 'data' attribute dictionary.  Optionally filters
        ui.json fields other than 'value'.
    """

    def __init__(
        self,
        ui_json: dict[str, Any] | UIJson,
    ):
        logger.warning(
            "The 'InputFile' class will be deprecated in future releases. "
            "Use the 'uijson.UIJson' class instead."
        )
        self._geoh5 = None
        self._data = None
        self._ui_json = self.validate_ui_json(ui_json)

    @property
    def data(self) -> dict[str, Any]:
        """
        Dictionary representing the input data for the ui.json file.
        """
        return self.ui_json.to_params()

    @staticmethod
    def read_ui_json(json_file: str | Path) -> InputFile:
        """
        Read and create an InputFile from ui.json
        """

        json_file_path = Path(json_file).resolve()
        ui_json = UIJson.read(json_file_path)

        return InputFile(ui_json)

    @property
    def ui_json(self) -> UIJson:
        """
        Dictionary representing the ui.json file with promoted values.
        """
        return self._ui_json

    def validate_ui_json(self, val: dict[str, Any] | UIJson) -> UIJson:
        if not isinstance(val, (dict, UIJson)):
            raise TypeError(
                f"Expected a dictionary or UIJson instance, got {type(val).__name__} instead."
            )

        if isinstance(val, dict):
            val = UIJson.from_dict(val)

        return val

    def update_ui_values(self, data: dict):
        """
        Update the ui.json values and enabled status from input data.

        :param data: Key and value pairs expected by the ui_json.

        :raises AttributeError: If attempting to set None value to non-optional parameter.
        """
        self.ui_json.set_values(**data)

    @property
    def geoh5(self):
        """Geoh5 workspace."""
        if self._geoh5 is None:
            self._geoh5 = self.data["geoh5"]
        return self._geoh5

    def write_ui_json(
        self,
        name: str | None = None,
        path: str | Path | None = None,
    ):
        """
        Writes a formatted ui.json file from InputFile data

        :param name: Name of the file
        :param path: Directory to write the ui.json to.
        """

        if name is None:
            name = "input_file.ui.json"

        if path is None:
            raise AttributeError(
                "The input file requires 'path' to be set before writing out."
            )

        if self.ui_json is None:
            raise AttributeError(
                "The input file requires 'ui_json' and 'data' to be set before writing out."
            )

        return self.ui_json.write(Path(path) / name)
