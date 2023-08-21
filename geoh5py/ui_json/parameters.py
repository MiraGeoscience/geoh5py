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

from typing import Any, TypeAlias
from uuid import UUID

from geoh5py.shared.exceptions import (
    AggregateValidationError,
    BaseValidationError,
    UIJsonFormatError,
)
from geoh5py.ui_json.validation import Validations

Validation: TypeAlias = dict[str, Any]


class Parameter:
    def __init__(self, name, value=None, validations: Validation | None = None):
        self.name: str = name
        self.value: Any = value
        self._validations: Validations = Validations(validations)

    @property
    def validations(self):
        return self._validations

    @validations.setter
    def validations(self, val):
        self._validations = Validations(val)

    def validate(self):
        self.validations.validate(self.name, self.value)

    def __str__(self):
        return f"<{type(self).__name__}> : '{self.name}' -> {self.value}"


class FormParameter:
    form_validations: dict[str, Validation] = {
        "label": {"required": True, "types": [str]},
        "value": {"required": True},
        "optional": {"types": [bool, type(None)]},
        "enabled": {"types": [bool, type(None)]},
        "main": {"types": [bool, type(None)]},
        "group": {"types": [str, type(None)]},
        "groupOptional": {"types": [bool, type(None)]},
        "dependency": {"types": [str, type(None)]},
        "dependencyType": {"values": ["enabled", "disabled", "show", "hide"]},
        "groupDependency": {"types": [str, type(None)]},
        "groupDependencyType": {"values": ["enabled", "disabled", "show", "hide"]},
        "tooltip": {"types": [str, type(None)]},
    }
    valid_members: list[str] = list(form_validations.keys())
    identifier_members: list[str] = []

    def __init__(
        self,
        name: str,
        form: dict[str, Any] | None = None,
        validations: Validation | None = None,
    ):
        self.name= name
        if form is not None:
            self._active_members = list(form)
            self.members = self._members(form)
        self._value._validations = Validations(validations)

    @property
    def validations(self) -> Validation | None:
        return self._value.validations

    @validations.setter
    def validations(self, val):
        self._value.validations = val

    @property
    def form(self):
        form = {}
        for member in self._active_members:
            if member == "value":
                form[member] = self.value
            else:
                form[member] = self.members[member].value
        return form

    @property
    def value(self) -> Any:
        return self._value.value

    @value.setter
    def value(self, val):
        self._value.value = val

    def validate(self, level: str = "form"):
        if level == "form":
            self._validate_form()
        elif level == "value":
            self._validate_value()
        elif level == "all":
            error_list = []
            for validate_method in ["_validate_value", "_validate_form"]:
                try:
                    getattr(self, validate_method)()
                except BaseValidationError as err:
                    error_list.append(err)

            if len(error_list) > 1:
                raise AggregateValidationError(self.name, error_list)
        else:
            raise ValueError(
                "Argument 'level' must be one of: 'all', 'form', or 'value'."
            )

    def _validate_value(self):
        self._value.validate()

    def _validate_form(self):
        for parameter in self.members.values():
            try:
                parameter.validate()
            except BaseValidationError as err:
                raise UIJsonFormatError(self.name, str(err)) from err

    def _members(self, form: dict[str, Any]):
        members = {}
        for member in self.valid_members:
            if member == "value":
                self._value = Parameter("value", form.get("value", None))
            else:
                value = form.get(member, None)
                members[member] = Parameter(
                    member, value, self.form_validations[member]
                )

        unrecognized_members = [k for k in form if k not in self.valid_members]
        for member in unrecognized_members:
            members[member] = Parameter(member, form[member])

        return members

    @classmethod
    def is_form(cls, form: dict[str, Any]) -> bool:
        id_members = cls.identifier_members
        return any(k in form for k in id_members)

    def __str__(self):
        return f"<{type(self).__name__}> : '{self.name}' -> {self.value}"

    def __getattr__(self, name):
        if name in self.valid_members:
            ret = self.members[name].value
        elif name in self.__dict__:
            ret = self.__dict__[name]
        else:
            raise AttributeError(f"Attribute '{name}' not a valid attribute.")

        return ret.value if isinstance(ret, Parameter) else ret


class StringParameter(FormParameter):
    base_validations: Validation = {"types": [str]}

    def __init__(self, name, form, validations=None):
        validations = (
            dict(self.base_validations, **validations)
            if validations
            else self.base_validations
        )
        super().__init__(name, form, validations)


class BoolParameter(FormParameter):
    base_validations: Validation = {"types": [bool]}

    def __init__(self, name, form, validations=None):
        validations = (
            dict(self.base_validations, **validations)
            if validations
            else self.base_validations
        )
        super().__init__(name, form, validations)


class IntegerParameter(FormParameter):
    base_validations: Validation = {"types": [int]}
    integer_form_validations: dict[str, Validation] = {
        "min": {"types": [int, type(None)]},
        "max": {"types": [int, type(None)]},
    }
    form_validations: dict[str, Validation] = dict(
        FormParameter.form_validations, **integer_form_validations
    )
    valid_members: list[str] = list(form_validations.keys())
    identifier_members: list[str] = []

    def __init__(self, name, form, validations=None):
        validations = (
            dict(self.base_validations, **validations)
            if validations
            else self.base_validations
        )
        super().__init__(name, form, validations)


class FloatParameter(FormParameter):
    base_validations: Validation = {"types": [float]}
    float_validations: dict[str, Validation] = {
        "min": {"types": [float, type(None)]},
        "max": {"types": [float, type(None)]},
        "precision": {"types": [int, type(None)]},
        "lineEdit": {"types": [bool, type(None)]},
    }
    form_validations: dict[str, Validation] = dict(
        FormParameter.form_validations, **float_validations
    )
    valid_members: list[str] = list(form_validations.keys())
    identifier_members: list[str] = ["precision", "lineEdit"]

    def __init__(self, name, form, validations=None):
        validations = (
            dict(self.base_validations, **validations)
            if validations
            else self.base_validations
        )
        super().__init__(name, form, validations)


class ChoiceStringParameter(FormParameter):
    base_validations: Validation = {"types": [str]}
    choice_string_validations: dict[str, Validation] = {
        "choiceList": {"required": True, "types": [list]}
    }
    form_validations: dict[str, Validation] = dict(
        FormParameter.form_validations, **choice_string_validations
    )
    valid_members: list[str] = list(form_validations.keys())
    identifier_members: list[str] = ["choiceList"]

    def __init__(self, name, form, validations=None):
        validations = (
            dict(self.base_validations, **validations)
            if validations
            else self.base_validations
        )
        super().__init__(name, form, validations)

    @property
    def choiceList(self):  # pylint: disable=invalid-name
        return self._choiceList

    @choiceList.setter
    def choiceList(self, val):  # pylint: disable=invalid-name
        self._choiceList = val  # pylint: disable=invalid-name

        if isinstance(val, Parameter):
            val = val.value

        if self.validations:
            self.validations.update({"values": val})
        else:
            self.validations = {"values": val}


class FileParameter(FormParameter):
    base_validations: Validation = {"types": [str]}
    file_validations: dict[str, Validation] = {
        "fileDescription": {"required": True, "types": [str, tuple, list]},
        "fileType": {"required": True, "types": [str, tuple, list]},
        "fileMulti": {"types": [bool, type(None)]},
    }
    form_validations: dict[str, Validation] = dict(
        FormParameter.form_validations, **file_validations
    )
    valid_members: list[str] = list(form_validations.keys())
    identifier_members: list[str] = ["fileDescription", "fileType", "fileMulti"]

    def __init__(self, name, form, validations=None):
        validations = (
            dict(self.base_validations, **validations)
            if validations
            else self.base_validations
        )
        super().__init__(name, form, validations)


class ObjectParameter(FormParameter):
    base_validations: Validation = {"types": [str, UUID]}
    object_validations: dict[str, Validation] = {
        "meshType": {
            "required": True,
            "values": [
                "{202C5DB1-A56D-4004-9CAD-BAAFD8899406}",
                "{6A057FDC-B355-11E3-95BE-FD84A7FFCB88}",
                "{F26FEBA3-ADED-494B-B9E9-B2BBCBE298E1}",
                "{48F5054A-1C5C-4CA4-9048-80F36DC60A06}",
                "{b020a277-90e2-4cd7-84d6-612ee3f25051}",
                "{4ea87376-3ece-438b-bf12-3479733ded46}",
            ],
        }
    }
    form_validations: dict[str, Validation] = dict(
        FormParameter.form_validations, **object_validations
    )
    valid_members: list[str] = list(form_validations.keys())
    identifier_members: list[str] = ["meshType"]

    def __init__(self, name, form, validations=None):
        validations = (
            dict(self.base_validations, **validations)
            if validations
            else self.base_validations
        )
        super().__init__(name, form, validations)


class DataParameter(FormParameter):
    base_validations: Validation = {"types": [str, UUID, type(None)]}
    data_validations: dict[str, Validation] = {
        "parent": {"required": True, "types": [str]},
        "association": {"required": True, "values": ["Vertex", "Cell"]},
        "dataType": {"required": True, "values": ["Float", "Integer", "Reference"]},
        "dataGroupType": {
            "values": [
                "3D vector",
                "Dip direction & dip",
                "Strike & dip",
                "Multi-element",
            ]
        },
    }
    form_validations: dict[str, Validation] = dict(
        FormParameter.form_validations, **data_validations
    )
    valid_members: list[str] = list(form_validations.keys())
    identifier_members: list[str] = ["dataGroupType"]

    def __init__(self, name, form, validations=None):
        validations = (
            dict(self.base_validations, **validations)
            if validations
            else self.base_validations
        )
        super().__init__(name, form, validations)


class DataValueParameter(FormParameter):
    base_validations: Validation = {"types": [int, float]}
    data_value_validations: dict[str, Validation] = {
        "parent": {"required": True, "types": [str]},
        "association": {"required": True, "values": ["Vertex", "Cell"]},
        "dataType": {"required": True, "values": ["Float", "Integer", "Reference"]},
        "isValue": {"required": True, "types": [bool]},
        "property": {"required": True, "types": [str, UUID, type(None)]},
    }
    form_validations: dict[str, Validation] = dict(
        FormParameter.form_validations, **data_value_validations
    )
    valid_members: list[str] = list(form_validations.keys())
    identifier_members: list[str] = ["isValue", "property"]

    def __init__(self, name, form, validations=None):
        validations = (
            dict(self.base_validations, **validations)
            if validations
            else self.base_validations
        )
        super().__init__(name, form, validations)

    @property
    def value(self):
        val = self.property
        if self.isValue:
            val = self._value.value
        return val

    @value.setter
    def value(self, val):
        if isinstance(val, (int, float)):
            self._value.value = val
            self.isValue = True  # pylint: disable=invalid-name
        else:
            self.property = val
            self.isValue = False

    def _validate_value(self):
        if self.isValue:
            self._value.validate()
        else:
            self.members["property"].validate()

    def _validate_form(self):
        for parameter in self.members.values():
            try:
                parameter.validate()
            except BaseValidationError as err:
                raise UIJsonFormatError(self.name, str(err)) from err
