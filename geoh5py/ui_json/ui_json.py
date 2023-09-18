#  Copyright (c) 2023 Mira Geoscience Ltd.
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

from typing import Any

from geoh5py.ui_json.descriptors import ValueAccess
from geoh5py.ui_json.enforcers import EnforcerPool
from geoh5py.ui_json.forms import FormParameter
from geoh5py.ui_json.parameters import Parameter


class UIJson:
    """Stores parameters and data for applications executed with ui.json files."""

    validations = {
        "required_uijson_parameters": [
            "title",
            "geoh5",
            "run_command",
            "run_command_boolean",
            "monitoring_directory",
            "conda_environment",
            "conda_environment_boolean",
            "workspace",
        ]
    }

    def __init__(self, parameters: dict[str, Parameter | FormParameter]):
        self.parameters: dict[str, Parameter | FormParameter] = parameters
        self.enforcers: EnforcerPool = EnforcerPool.from_validations(
            self.name, self.validations
        )
        self._allow_values_access()

    def _allow_values_access(self):
        """Public parameter names access underlying parameter value."""
        for parameter in self.parameters:
            if parameter not in dir(self):
                setattr(self, f"_{parameter}", self.parameters[parameter])
                setattr(self.__class__, parameter, ValueAccess(f"_{parameter}"))

    def to_dict(self, naming: str = "snake") -> dict[str, Any]:
        """
        Returns a dictionary of name and value/form for each parameter.

        :param naming: Uses MEMBER_KEYS.map to convert python names to
            camel case for writing to file.
        """

        use_camel = naming == "camel"

        def get_data(param: Parameter | FormParameter):
            return param.form(use_camel) if hasattr(param, "form") else param.value

        out = {}
        for param, value in self.parameters.items():
            out[param] = get_data(value)

        return out

    def update(self, data: dict[str, Any]):
        for param, value in data.items():
            if param in self.parameters:
                setattr(self, param, value)

    def validate(self):
        """Validates uijson data against a pool of enforcers."""
        uijson = self.to_dict()
        self.enforcers.enforce(uijson)

    @property
    def name(self) -> str:
        """Returns a name for the uijson file."""

        uijson = self.to_dict()
        name = "uijson"
        if "title" in uijson:
            name = uijson["title"]
        elif "geoh5" in uijson:
            name = uijson["geoh5"].h5file.stem

        return name
