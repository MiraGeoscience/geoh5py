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
    def __init__(self, name, value, validations=None):
        self.name: str = name
        self.value: Any = value
        self._validations: Validations | None = (
            Validations(validations) if isinstance(validations, dict) else validations
        )

    @property
    def validations(self) -> Validation | None:
        if self._validations is None:
            out = None
        else:
            out = self._validations.validations

        return out

    @validations.setter
    def validations(self, val):
        if val is None:
            self._validations = None
        else:
            self._validations = (
                val if isinstance(val, Validations) else Validations(val)
            )

    def validate(self):
        if self._validations is None:
            msg = "Must set validations before calling validate()."
            raise AttributeError(msg)
        self._validations.validate(self.name, self.value)

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

    def __init__(self, name, form, validations: Validations | Validation | None = None):
        self.name: str = name
        self._form: dict[str, Any] = form
        self._update_attrs(form)

        self.validations: Validation | Validations | None = (
            Validations(validations)  # type: ignore
            if isinstance(validations, dict)
            else validations
        )

    @property
    def validations(self) -> Validation | None:
        if self._value.validations is None:
            out = None
        else:
            out = self._value.validations

        return out

    @validations.setter
    def validations(self, val):
        if val is None:
            self._validations = None
        elif self._value.validations is not None and isinstance(self.validations, dict):
            val = val if isinstance(val, dict) else val.validations
            self._value.validations = dict(self.validations, **val)
        else:
            self._value.validations = val

    @property
    def value(self) -> Any:
        return self._value.value

    @value.setter
    def value(self, val):
        self._value = Parameter("value", val, self.validations)

    @property
    def form(self) -> dict[str, Any]:
        form_assembly = {}
        for name in self.valid_members:
            if name == "value":
                member = self._value.value
            else:
                member = getattr(self, name).value

            is_required = self.form_validations[name].get("required", False)
            if member is None and not is_required:
                continue

            form_assembly[name] = member

        return form_assembly

    @form.setter
    def form(self, val: dict[str, Any]):
        self._update_attrs(val)

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
        for member in [k for k in self.valid_members if k != "value"]:
            try:
                getattr(self, member).validate()
            except BaseValidationError as err:
                raise UIJsonFormatError(self.name, str(err)) from err

    def _update_attrs(self, form: dict[str, Any]):
        for member in self.valid_members:
            if member == "value":
                self._value = Parameter("value", form.get("value", None))
            else:
                value = form.get(member, None)
                setattr(
                    self,
                    member,
                    Parameter(member, value, self.form_validations[member]),
                )

    @classmethod
    def is_form(cls, form: dict[str, Any]) -> bool:
        id_members = cls.identifier_members
        return any(k in form for k in id_members)

    def __str__(self):
        return f"<{type(self).__name__}> : '{self.name}' -> {self.value}"


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
        val = self.property.value
        if self.isValue.value:
            val = self._value.value
        return val

    @value.setter
    def value(self, val):
        if isinstance(val, (int, float)):
            self.property = Parameter("property", val, self.validations)
            self.isValue = Parameter(  # pylint: disable=invalid-name
                "isValue", False, self.form_validations["isValue"]
            )
        else:
            self._value = Parameter("value", val, self.validations)
            self.isValue = Parameter("isValue", True, self.form_validations["isValue"])

    def _validate_value(self):
        if self.isValue:
            self._value.validate()
        else:
            self.property.validate()

    def _validate_form(self):
        for member in [k for k in self.valid_members if k not in ["value", "property"]]:
            try:
                getattr(self, member).validate()
            except BaseValidationError as err:
                raise UIJsonFormatError(self.name, str(err)) from err
