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

from geoh5py.ui_json.parameters import FormParameter, Parameter

Parameters = list[Parameter | FormParameter]


class UIJson:
    """Stores parameters and data for applications executed with ui.json files."""

    def __init__(self, parameters):
        self.parameters: Parameters = parameters

    def to_dict(self):
        """Returns a dictionary of key and value/form for each parameter."""

        def get_data(param):
            return param.value if isinstance(param, Parameter) else param.form

        return {k.name: get_data(k) for k in self.parameters}
