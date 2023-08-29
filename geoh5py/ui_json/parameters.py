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

from typing import Any, Dict
from uuid import UUID

import numpy as np

from geoh5py.shared.exceptions import AggregateValidationError, BaseValidationError
from geoh5py.ui_json.validation import Validations

Validation = Dict[str, Any]

KEY_MAP = {
    "groupOptional": "group_optional",
    "dependencyType": "dependency_type",
    "groupDependency": "group_dependency",
    "groupDependencyType": "group_dependency_type",
    "lineEdit": "line_edit",
    "choiceList": "choice_list",
    "fileDescription": "file_description",
    "fileType": "file_type",
    "fileMulti": "file_multi",
    "meshType": "mesh_type",
    "dataType": "data_type",
    "dataGroupType": "data_group_type",
    "dataType": "data_type",
    "isValue": "is_value",
}


class Parameter:
    """
    Basic parameter to store key/value data with validation capabilities.

    :param name: Parameter name.
    :param value: The parameters value.
    :param validations: Parameter validations
    """

    def __init__(self, name, value=None, validations: Validation | None = None):
        self._validations: Validations = Validations(validations)
        self.name: str = name
        setattr(self, "_value" if value is None else "value", value)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self._value = val
        if self.validations:
            self.validate()

    @property
    def validations(self) -> Validations:
        return self._validations

    @validations.setter
    def validations(self, val):
        self._validations = Validations(val)

    def validate(self):
        self.validations.validate(self.name, self.value)

    def __str__(self):
        return f"<{type(self).__name__}> : '{self.name}' -> {self.value}"


class ValueAccess:
    """
    Descriptor to elevate underlying member values within 'FormParameter'.

    :param private: Name of private attribute.
    """

    def __init__(self, private: str):
        self.private = private

    def __get__(self, obj, objtype=None):
        return getattr(obj, self.private).value

    def __set__(self, obj, value):
        setattr(getattr(obj, self.private), "value", value)
        obj._active_members.append(self.private[1:])


class FormParameter:
    """
    Base class for parameters that create visual ui elements from a form.

    :param name: Parameter name.
    :param value: The parameters value.
    :param validations: Parameter validations
    :param form_validations: Parameter's form validations
    :param form: dictionary specifying visual characteristics of a ui element.
    :param active: list of form members to include in form.
    :param required: list of form members that must be included to create a
        valid form.

    :note: Can be constructed from keyword arguments of through the
        'from_dict' constructor.

    :note: The form members may be updated with a dictionary of members and
        associated data through the 'register' method.

    :note: Standardized form members (in valid_members list) will also
        be accessible through a descriptor that sets/gets the underlying
        values attribute of the private Parameter object.
    """

    form_validations: dict[str, Validation] = {
        "label": {"required": True, "types": [str]},
        "value": {"required": True},
        "enabled": {"types": [bool, type(None)]},
        "optional": {"types": [bool, type(None)]},
        "main": {"types": [bool, type(None)]},
        "group": {"types": [str, type(None)]},
        "group_optional": {"types": [bool, type(None)]},
        "dependency": {"types": [str, type(None)]},
        "dependency_type": {"values": ["enabled", "disabled", "show", "hide"]},
        "group_dependency": {"types": [str, type(None)]},
        "group_dependency_type": {"values": ["enabled", "disabled", "show", "hide"]},
        "tooltip": {"types": [str, type(None)]},
    }
    valid_members: list[str] = list(form_validations.keys())
    identifier_members: list[str] = []

    def __init__(self, name: str, validations: Validation | None = None, **kwargs):
        self.name: str = name
        self._value = Parameter("value", None)
        self._label: str | None = None
        self._enabled: bool = True
        self._optional: bool = False
        self._group_optional: bool = False
        self._main: bool = True
        self._group: str | None = None
        self._dependency: str | None = None
        self._dependency_type: str | None = None
        self._group_dependency: str | None = None
        self._group_dependency_type: str | None = None
        self._tooltip: str | None = None
        self._extra_members: dict[str, Any] = {}
        self._active_members: list[str] = list(kwargs)
        self.validations: Validations = Validations(validations)
        self.register(kwargs)

    @classmethod
    def from_dict(
        cls, name: str, form: dict[str, Any], validations: dict | None = None
    ):
        return cls(name, validations, **form)

    @property
    def value(self):
        return self._value.value

    @value.setter
    def value(self, val):
        self._value.value = val

    def register(self, members: dict[str, Any]):
        """
        Set parameters from form members with default or incoming values.

        :param members: Dictionary of form members and associated data.

        :return: Dictionary of unrecognized members and data.
        """

        if not members:
            return

        if not isinstance(members, dict):
            raise TypeError("Input 'members' must be a dictionary.")

        members = {KEY_MAP.get(k, k): v for k, v in members.items()}
        error_list = []
        for member in self.valid_members:
            validations = (
                self.validations if member == "value" else self.form_validations[member]
            )
            if member in members:
                try:
                    param = Parameter(
                        member, members.pop(member), validations  # type: ignore
                    )
                except BaseValidationError as err:
                    error_list.append(err)
            else:
                param = Parameter(member, validations=validations)  # type: ignore

            setattr(self, f"_{member}", param)
            if member not in dir(self):  # do not override pre-defined properties
                setattr(self.__class__, member, ValueAccess(f"_{member}"))

        if error_list:
            if len(error_list) == 1:
                raise error_list.pop()

            raise AggregateValidationError(self.name, error_list)

        self._extra_members.update(members)

    @property
    def validations(self) -> Validations:
        return self._value.validations

    @validations.setter
    def validations(self, val):
        self._value.validations = val

    @property
    def active(self) -> list[str]:
        """
        Returns names of active form members.

        :return: List of active form members.  These will include any members
            that were:
                1. Provided during construction.
                2. Updated through the 'register' method.
                3. Updated through member setters.
                4. Defined as 'required' by the validations.
        """
        active = self._active_members + list(self._extra_members)
        active_unique, ind = np.unique(active, return_index=True)
        return list(active_unique[ind])  # Preserve order after unique

    @property
    def required(self):
        """Returns list of members required by their validations."""
        return [k for k, v in self.form_validations.items() if v.get("required", False)]

    @property
    def form(self):
        """Returns dictionary of active form members and their values."""
        form = {}
        for member in self.active:
            if member in self._extra_members:
                form[member] = self._extra_members[member]
            else:
                form[member] = getattr(self, member)

        return form

    @classmethod
    def is_form(cls, form: dict[str, Any]) -> bool:
        """Returns True if form contains any identifier members."""
        id_members = cls.identifier_members
        form_members = [KEY_MAP.get(k, k) for k in form]
        return any(k in form_members for k in id_members)

    def __str__(self):
        return f"<{type(self).__name__}> : '{self.name}' -> {self.value}"


class StringParameter(FormParameter):
    """String parameter type."""

    base_validations: Validation = {"types": [str]}

    def __init__(self, name, validations=None, **kwargs):
        validations = (
            dict(self.base_validations, **validations)
            if validations
            else self.base_validations
        )
        super().__init__(name, validations=validations, **kwargs)


class BoolParameter(FormParameter):
    """Boolean parameter type."""

    base_validations: Validation = {"types": [bool]}

    def __init__(self, name, validations=None, **kwargs):
        validations = (
            dict(self.base_validations, **validations)
            if validations
            else self.base_validations
        )
        super().__init__(name, validations=validations, **kwargs)


class IntegerParameter(FormParameter):
    """Integer parameter type."""

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

    def __init__(self, name, validations=None, **kwargs):
        self._min: int | None = None
        self._max: int | None = None
        validations = (
            dict(self.base_validations, **validations)
            if validations
            else self.base_validations
        )
        super().__init__(name, validations=validations, **kwargs)


class FloatParameter(FormParameter):
    """Float parameter type."""

    base_validations: Validation = {"types": [float]}
    float_validations: dict[str, Validation] = {
        "min": {"types": [float, type(None)]},
        "max": {"types": [float, type(None)]},
        "precision": {"types": [int, type(None)]},
        "line_edit": {"types": [bool, type(None)]},
    }
    form_validations: dict[str, Validation] = dict(
        FormParameter.form_validations, **float_validations
    )
    valid_members: list[str] = list(form_validations.keys())
    identifier_members: list[str] = ["precision", "line_edit"]

    def __init__(self, name, validations=None, **kwargs):
        self._min: float | None = None
        self._max: float | None = None
        self._precision: int | None = None
        self._line_edit: bool | None = None
        validations = (
            dict(self.base_validations, **validations)
            if validations
            else self.base_validations
        )
        super().__init__(name, validations=validations, **kwargs)


class ChoiceStringParameter(FormParameter):
    """Choice string parameter type."""

    base_validations: Validation = {"types": [str]}
    choice_string_validations: dict[str, Validation] = {
        "choice_list": {"required": True, "types": [list]}
    }
    form_validations: dict[str, Validation] = dict(
        FormParameter.form_validations, **choice_string_validations
    )
    valid_members: list[str] = list(form_validations.keys())
    identifier_members: list[str] = ["choice_list"]

    def __init__(self, name, validations=None, **kwargs):
        self._choice_list: list | None = None
        validations = (
            dict(self.base_validations, **validations)
            if validations
            else self.base_validations
        )
        super().__init__(name, validations=validations, **kwargs)

    @property
    def choice_list(self):
        return self._choice_list

    @choice_list.setter
    def choice_list(self, val):
        self._choice_list = val

        if isinstance(val, Parameter):
            val = val.value

        if self.validations:
            self.validations.update({"values": val})
        else:
            self.validations = {"values": val}  # type: ignore


class FileParameter(FormParameter):
    """File parameter type."""

    base_validations: Validation = {"types": [str]}
    file_validations: dict[str, Validation] = {
        "file_description": {"required": True, "types": [str, tuple, list]},
        "file_type": {"required": True, "types": [str, tuple, list]},
        "file_multi": {"types": [bool, type(None)]},
    }
    form_validations: dict[str, Validation] = dict(
        FormParameter.form_validations, **file_validations
    )
    valid_members: list[str] = list(form_validations.keys())
    identifier_members: list[str] = ["file_description", "file_type", "file_multi"]

    def __init__(self, name, validations=None, **kwargs):
        self._file_description: str | tuple | list | None = None
        self._file_type: str | tuple | list | None = None
        self._file_multi: bool = False
        validations = (
            dict(self.base_validations, **validations)
            if validations
            else self.base_validations
        )
        super().__init__(name, validations=validations, **kwargs)


class ObjectParameter(FormParameter):
    """Object parameter type."""

    base_validations: Validation = {"types": [str, UUID]}
    object_validations: dict[str, Validation] = {
        "mesh_type": {
            "required": True,
            "types": [str, UUID, list],
        }
    }
    form_validations: dict[str, Validation] = dict(
        FormParameter.form_validations, **object_validations
    )
    valid_members: list[str] = list(form_validations.keys())
    identifier_members: list[str] = ["mesh_type"]

    def __init__(self, name, validations=None, **kwargs):
        self._mesh_type: list = []
        validations = (
            dict(self.base_validations, **validations)
            if validations
            else self.base_validations
        )
        super().__init__(name, validations=validations, **kwargs)


class DataParameter(FormParameter):
    """Data parameter type."""

    base_validations: Validation = {"types": [str, UUID, type(None)]}
    data_validations: dict[str, Validation] = {
        "parent": {"required": True, "types": [str, UUID]},
        "association": {"required": True, "values": ["Vertex", "Cell"]},
        "data_type": {"required": True, "values": ["Float", "Integer", "Reference"]},
        "data_group_type": {
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
    identifier_members: list[str] = ["data_group_type"]

    def __init__(self, name, validations=None, **kwargs):
        self._parent: str | UUID | None = None
        self._association: str | None = None
        self._data_type: str | None = None
        self._data_group_type: str | None = None
        validations = (
            dict(self.base_validations, **validations)
            if validations
            else self.base_validations
        )
        super().__init__(name, validations=validations, **kwargs)


class DataValueParameter(FormParameter):
    """Data value parameter type."""

    base_validations: Validation = {"types": [int, float]}
    data_value_validations: dict[str, Validation] = {
        "parent": {"required": True, "types": [str, UUID]},
        "association": {"required": True, "values": ["Vertex", "Cell"]},
        "data_type": {"required": True, "values": ["Float", "Integer", "Reference"]},
        "is_value": {"required": True, "types": [bool]},
        "property": {"required": True, "types": [str, UUID]},
    }
    form_validations: dict[str, Validation] = dict(
        FormParameter.form_validations, **data_value_validations
    )
    valid_members: list[str] = list(form_validations.keys())
    identifier_members: list[str] = ["is_value", "property"]

    def __init__(self, name, validations=None, **kwargs):
        self._parent: str | UUID | None = None
        self._association: str | None = None
        self._data_type: str | None = None
        self._is_value: bool | None = None
        self._property: str | UUID | None = None
        validations = (
            dict(self.base_validations, **validations)
            if validations
            else self.base_validations
        )
        super().__init__(name, validations=validations, **kwargs)

    @property
    def value(self):
        val = self.property
        if self.is_value:
            val = self._value.value
        return val

    @value.setter
    def value(self, val):
        if isinstance(val, (int, float)):
            self._value.value = val
            self.is_value = True
        else:
            self.property = val
            self.is_value = False
