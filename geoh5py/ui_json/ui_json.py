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

from geoh5py.ui_json.forms import FormParameter
from geoh5py.ui_json.parameters import Parameter


class UIJson:  # pylint: disable=too-few-public-methods
    """Stores parameters and data for applications executed with ui.json files."""

    def __init__(self, parameters):
        self.parameters: list[Parameter | FormParameter] = parameters

    def to_dict(self, naming="snake"):
        """
        Returns a dictionary of key and value/form for each parameter.

        :param naming: Uses KEY_MAP to convert python names to camel case
            for writing to file.
        """

        def get_data(param):
            return param.form(naming) if hasattr(param, "form") else param.value

        out = {}
        for param in self.parameters:
            out[param.name] = get_data(param)

        return out
