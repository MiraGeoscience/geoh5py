#  Copyright (c) 2022 Mira Geoscience Ltd.
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

# pylint: disable=R0913

from __future__ import annotations

import inspect
from typing import Any
from uuid import UUID

from geoh5py.shared.exceptions import (
    AggregateValidationError,
    BaseValidationError,
    UIJsonFormatError,
)
from geoh5py.ui_json.validation import Validations

from .. import objects
from ..shared import Entity

known_types = [
    member.default_type_uid()
    for _, member in inspect.getmembers(objects)
    if hasattr(member, "default_type_uid") and member.default_type_uid() is not None
]


class Parameter:
    def __init__(self, name, value, validations):
        self.name: str = name
        self.value: Any = value
        self.validations: dict[str, dict] | Validations = validations

    @property
    def validations(self):
        return self._validations.validations

    @validations.setter
    def validations(self, val):
        if hasattr(self, "_validations"):
            self._validations.validations = dict(self.validations, **val)
        else:
            self._validations = (
                val if isinstance(val, Validations) else Validations(val)
            )

    def validate(self):
        self._validations.validate(self.name, self.value)

    def __str__(self):
        return f"<{type(self).__name__}> : '{self.name}' -> {self.value}"


class FormParameter:

    form_validations = {
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
    valid_members = list(form_validations.keys())
    identifier_members = []

    def __init__(self, name, form, validations):

        self.name: str = name
        self.validations: dict[str, dict] | Validations = validations
        self.form: dict[str, Any] = form

    @property
    def value(self):
        return self._value.value

    @value.setter
    def value(self, val):
        self._value = Parameter("value", val, self.validations)

    @property
    def form(self):

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
    def form(self, val):

        for member in self.valid_members:
            if member == "value":
                self._value = Parameter(
                    "value", val.get("value", None), self.validations
                )
            else:
                value = val.get(member, None)
                setattr(
                    self,
                    member,
                    Parameter(member, value, self.form_validations[member]),
                )

    def _validate_value(self):
        self._value.validate()

    def _validate_form(self):
        for member in [k for k in self.valid_members if k != "value"]:
            try:
                getattr(self, member).validate()
            except BaseValidationError as err:
                raise UIJsonFormatError(self.name, str(err))

    def validate(self, level="form"):

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

    @classmethod
    def is_form(cls, form):
        id_members = cls.identifier_members
        return any(k in form for k in id_members)

    def __str__(self):
        return f"<{type(self).__name__}> : '{self.name}' -> {self.value}"


class StringParameter(FormParameter):

    base_validations = {"types": [str]}

    def __init__(self, name, form, validations):
        super().__init__(name, form, dict(self.base_validations, **validations))


class BoolParameter(FormParameter):

    base_validations = {"types": [bool]}

    def __init__(self, name, form, validations):
        super().__init__(name, form, dict(self.base_validations, **validations))


class IntegerParameter(FormParameter):

    base_validations = {"types": [int]}
    integer_form_validations = {
        "min": {"types": [int, type(None)]},
        "max": {"types": [int, type(None)]},
    }
    form_validations = dict(FormParameter.form_validations, **integer_form_validations)
    valid_members = list(form_validations.keys())
    identifier_members = []

    def __init__(self, name, form, validations):
        super().__init__(name, form, dict(self.base_validations, **validations))


class FloatParameter(FormParameter):

    base_validations = {"types": [float]}
    float_validations = {
        "min": {"types": [float, type(None)]},
        "max": {"types": [float, type(None)]},
        "precision": {"types": [int, type(None)]},
        "lineEdit": {"types": [bool, type(None)]},
    }
    form_validations = dict(FormParameter.form_validations, **float_validations)
    valid_members = list(form_validations.keys())
    identifier_members = ["precision", "lineEdit"]

    def __init__(self, name, form, validations):
        super().__init__(name, form, dict(self.base_validations, **validations))


class ChoiceStringParameter(FormParameter):

    base_validations = {"types": [str]}
    choice_string_validations = {"choiceList": {"required": True, "types": [list]}}
    form_validations = dict(FormParameter.form_validations, **choice_string_validations)
    valid_members = list(form_validations.keys())
    identifier_members = ["choiceList"]

    def __init__(self, name, form, validations):
        super().__init__(name, form, dict(self.base_validations, **validations))

    @property
    def choiceList(self):
        return self._choiceList

    @choiceList.setter
    def choiceList(self, val):
        self.validations.update({"values": val})
        self._choiceList = val


class FileParameter(FormParameter):

    base_validations = {"types": [str]}
    file_validations = {
        "fileDescription": {"required": True, "types": [str, tuple, list]},
        "fileType": {"required": True, "types": [str, tuple, list]},
        "fileMulti": {"types": [bool]},
    }
    form_validations = dict(FormParameter.form_validations, **file_validations)
    valid_members = list(form_validations.keys())
    identifier_members = ["fileDescription", "fileType", "fileMulti"]

    def __init__(self, name, form, validations):
        super().__init__(name, form, dict(self.base_validations, **validations))


class ObjectParameter(FormParameter):

    base_validations = {"types": [str, UUID]}
    object_validations = {
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
    form_validations = dict(FormParameter.form_validations, **object_validations)
    valid_members = list(form_validations.keys())
    identifier_members = ["meshType"]

    def __init__(self, name, form, validations):
        super().__init__(name, form, dict(self.base_validations, **validations))


class DataParameter(FormParameter):

    base_validations = {"types": [str, UUID, type(None)]}
    data_validations = {
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
    form_validations = dict(FormParameter.form_validations, **data_validations)
    valid_members = list(form_validations.keys())
    identifier_members = ["dataGroupType"]

    def __init__(self, name, form, validations):
        super().__init__(name, form, dict(self.base_validations, **validations))


class DataValueParameter(FormParameter):

    base_validations = {"types": [int, float]}
    data_value_validations = {
        "parent": {"required": True, "types": [str]},
        "association": {"required": True, "values": ["Vertex", "Cell"]},
        "dataType": {"required": True, "values": ["Float", "Integer", "Reference"]},
        "isValue": {"required": True, "types": [bool]},
        "property": {"required": True, "types": [str, UUID, type(None)]},
    }
    form_validations = dict(FormParameter.form_validations, **data_value_validations)
    valid_members = list(form_validations.keys())
    identifier_members = ["isValue", "property"]

    def __init__(self, name, form, validations):
        super().__init__(name, form, dict(self.base_validations, **validations))

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
            self.isValue = Parameter("isValue", False, self.form_validations["isValue"])
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
                raise UIJsonFormatError(self.name, str(err))


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
            elif isinstance(value, dict):
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


def optional_parameter(state: str) -> dict[str, bool]:
    """
    Returns dictionary to make existing ui optional via .update() method.

    :param state: Initial state of optional parameter. Can be 'enabled' or 'disabled'.
    """

    var = {"optional": True}
    if state == "enabled":
        var["enabled"] = True
    elif state == "disabled":
        var["enabled"] = False
    else:
        raise ValueError(
            "Unrecognized state option. Must be either 'enabled' or 'disabled'."
        )

    return var


def bool_parameter(
    main: bool = True, label: str = "Logical data", value: bool = False
) -> dict:
    """
    Checkbox for true/false choice.

    :param main: Show ui in main.
    :param label: Label identifier.
    :param value: Input value.

    :returns: Ui_json compliant dictionary.
    """
    return {
        "main": main,
        "label": label,
        "value": value,
    }


def integer_parameter(
    main: bool = True,
    label: str = "Integer data",
    value: int = 1,
    vmin: int = 0,
    vmax: int = 100,
    optional: str | None = None,
) -> dict:
    """
    Input box for integer value.

    :param main: Show ui in main.
    :param label: Label identifier.
    :param value: Input value.
    :param vmin: Minimum value allowed.
    :param vmax: Maximum value allowed.
    :param optional: Make optional if not None. Initial state provided by not None
        value.  Can be either 'enabled' or 'disabled'.


    :returns: Ui_json compliant dictionary.
    """
    form = {"main": main, "label": label, "value": value, "min": vmin, "max": vmax}
    if optional is not None:
        form.update(optional_parameter(optional))
    return form


def float_parameter(
    main: bool = True,
    label: str = "Float data",
    value: float = 1.0,
    vmin: float = 0.0,
    vmax: float = 100.0,
    precision: int = 2,
    line_edit: bool = True,
    optional: str | None = None,
) -> dict:
    """
    Input box for float value.

    :param main: Show form in main.
    :param label: Label identifier.
    :param value: Input value.
    :param vmin: Minimum value allowed.
    :param vmax: Maximum value allowed.
    :param line_edit: Allow line edit or spin box
    :param optional: Make optional if not None. Initial state provided by not None
        value.  Can be either 'enabled' or 'disabled'.

    :returns: Ui_json compliant dictionary.
    """
    form = {
        "main": main,
        "label": label,
        "value": value,
        "min": vmin,
        "precision": precision,
        "lineEdit": line_edit,
        "max": vmax,
    }

    if optional is not None:
        form.update(optional_parameter(optional))
    return form


def string_parameter(
    main: bool = True,
    label: str = "String data",
    value: str = "data",
    optional: str | None = None,
) -> dict:
    """
    Input box for string value.

    :param main: Show form in main.
    :param label: Label identifier.
    :param value: Input string value.
    :param optional: Make optional if not None. Initial state provided by not None
        value.  Can be either 'enabled' or 'disabled'.

    :returns: Ui_json compliant dictionary.
    """
    form = {"main": main, "label": label, "value": value}

    if optional is not None:
        form.update(optional_parameter(optional))
    return form


def choice_string_parameter(
    main: bool = True,
    label: str = "String data",
    choice_list: tuple = ("Option A", "Option B"),
    value: str = "Option A",
    optional: str | None = None,
) -> dict:
    """
    Dropdown menu of string choices.

    :param main: Show form in main.
    :param label: Label identifier.
    :param value: Input value.
    :param choice_list: List of options.
    :param optional: Make optional if not None. Initial state provided by not None
        value.  Can be either 'enabled' or 'disabled'.

    :returns: Ui_json compliant dictionary.
    """
    form = {"main": main, "label": label, "value": value, "choiceList": choice_list}

    if optional is not None:
        form.update(optional_parameter(optional))
    return form


def file_parameter(
    main: bool = True,
    label: str = "File choices",
    file_description: tuple = (),
    file_type: tuple = (),
    value: str = "",
    optional: str | None = None,
) -> dict:
    """
    File loader for specific extensions.

    :param main: Show form in main.
    :param label: Label identifier.
    :param value: Input value.
    :param file_description: Title used to describe each type.
    :param file_type: Extension of files to display.
    :param optional: Make optional if not None. Initial state provided by not None
        value.  Can be either 'enabled' or 'disabled'.

    :returns: Ui_json compliant dictionary.
    """
    form = {
        "fileDescription": file_description,
        "fileType": file_type,
        "main": main,
        "label": label,
        "value": value,
    }

    if optional is not None:
        form.update(optional_parameter(optional))
    return form


def object_parameter(
    main: bool = True,
    label: str = "Object",
    mesh_type: tuple = tuple(known_types),
    value: str = None,
    optional: str | None = None,
) -> dict:
    """
    Dropdown menu of objects of specific types.

    :param main: Show form in main.
    :param label: Label identifier.
    :param value: Input value.
    :param mesh_type: Type of selectable objects.
    :param optional: Make optional if not None. Initial state provided by not None
        value.  Can be either 'enabled' or 'disabled'.

    :returns: Ui_json compliant dictionary.
    """
    form = {"main": main, "label": label, "value": value, "meshType": mesh_type}

    if optional is not None:
        form.update(optional_parameter(optional))
    return form


def data_parameter(
    main: bool = True,
    label: str = "Data channel",
    association: str = "Vertex",
    data_type: str = "Float",
    data_group_type: str = None,
    parent: str = "",
    value: str = "",
    optional: str | None = None,
) -> dict:
    """
    Dropdown menu of data from parental object.

    :param main: Show form in main.
    :param label: Label identifier.
    :param value: Input value.
    :param association: Data association type from 'Vertex' or 'Cell'.
    :param data_type: Type of data selectable from 'Float', 'Integer' or 'Reference'.
    :param data_group_type: [Optional] Select from property_groups of type.
        '3D vector',
        'Dip direction & dip',
        'Strike & dip',
        or 'Multi-element'.
    :param parent: Parameter name corresponding to the parent object.
    :param optional: Make optional if not None. Initial state provided by not None
        value.  Can be either 'enabled' or 'disabled'.

    :returns: Ui_json compliant dictionary.
    """
    form = {
        "main": main,
        "association": association,
        "dataType": data_type,
        "label": label,
        "parent": parent,
        "value": value,
    }

    if data_group_type is not None and data_group_type in [
        "3D vector",
        "Dip direction & dip",
        "Strike & dip",
        "Multi-element",
    ]:
        form["dataGroupType"] = data_group_type

    if optional is not None:
        form.update(optional_parameter(optional))

    return form


def data_value_parameter(
    main: bool = True,
    label: str = "Data channel",
    association: str = "Vertex",
    data_type: str = "Float",
    parent: str = "",
    value: float = 0.0,
    is_value: bool = True,
    prop: UUID | Entity | None = None,
    optional: str | None = None,
) -> dict:
    """
    Dropdown of data or input box.

    :param main: Show form in main.
    :param label: Label identifier.
    :param value: Input value.
    :param association: Data association type from 'Vertex' or 'Cell'.
    :param data_type: Type of data selectable from 'Float', 'Integer' or 'Reference'.
    :param data_group_type: [Optional] Select from property_groups of type.
        '3D vector',
        'Dip direction & dip',
        'Strike & dip',
        or 'Multi-element'.
    :param parent: Parameter name corresponding to the parent object.
    :param is_value: Display the input box or dropdown menu.
    :param prop: Data entity selected in the dropdown menu if 'is_value=False'.
    :param optional: Make optional if not None. Initial state provided by not None
        value.  Can be either 'enabled' or 'disabled'.

    :returns: Ui_json compliant dictionary.
    """
    form = {
        "main": main,
        "association": association,
        "dataType": data_type,
        "label": label,
        "parent": parent,
        "value": value,
        "isValue": is_value,
        "property": prop,
        "min": value,
    }

    if optional is not None:
        form.update(optional_parameter(optional))
    return form
