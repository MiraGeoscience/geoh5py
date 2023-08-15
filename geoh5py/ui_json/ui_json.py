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

from geoh5py.shared.exceptions import AggregateValidationError, BaseValidationError
from geoh5py.ui_json.parameters import (
    BoolParameter,
    FloatParameter,
    FormParameter,
    IntegerParameter,
    Parameter,
    StringParameter,
)


class UIJson:
    def __init__(self, parameters, validations=None):
        self.validations = {} if validations is None else validations
        self.parameters: dict[
            str, Parameter | FormParameter | dict[str, Any]
        ] = parameters

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, val):
        for name, value in val.items():
            if isinstance(value, (Parameter, FormParameter)):
                continue
            if isinstance(value, dict):
                parameter_class = UIJson.identify(value)
                val[name] = parameter_class(name, value, self.validations.get(name, {}))
            else:
                val[name] = Parameter(name, value, self.validations.get(name, {}))

        self._parameters = val

    @property
    def values(self):
        return {k: p.value for k, p in self.parameters.items()}

    @property
    def forms(self):
        val = {}
        for name, parameter in self.parameters.items():
            if isinstance(parameter, Parameter):
                val[name] = parameter.value
            else:
                val[name] = parameter.form
        return val

    def validate(self, level="form"):
        error_list = []
        for parameter in self.parameters.values():
            kwargs = {} if isinstance(parameter, Parameter) else {"level": level}
            try:
                parameter.validate(**kwargs)
            except BaseValidationError as err:
                error_list.append(err)

        if error_list:
            if len(error_list) == 1:
                raise error_list.pop()

            raise AggregateValidationError("test", error_list)

    @staticmethod
    def _parameter_class(parameter):
        found = FormParameter
        for candidate in FormParameter.__subclasses__():
            if candidate.is_form(parameter):
                found = candidate
        return found

    @staticmethod
    def _possible_parameter_classes(parameter):
        filtered_candidates = []
        candidates = FormParameter.__subclasses__()
        basic_candidates = [
            StringParameter,
            IntegerParameter,
            FloatParameter,
            BoolParameter,
        ]
        base_members = FormParameter.valid_members
        for candidate in candidates:
            possible_members = set(candidate.valid_members).difference(base_members)
            if any(k in possible_members for k in parameter):
                filtered_candidates.append(candidate)
        return filtered_candidates if filtered_candidates else basic_candidates

    @staticmethod
    def identify(parameter):
        winner = UIJson._parameter_class(parameter)
        if winner == FormParameter:
            possibilities = UIJson._possible_parameter_classes(parameter)
            n_candidates = len(possibilities)
            if n_candidates == 1:
                winner = possibilities[0]
            else:
                for candidate in possibilities:
                    try:
                        obj = candidate("test", parameter, {})
                        obj.validate("value")
                        winner = candidate
                    except BaseValidationError:
                        pass

        return winner
