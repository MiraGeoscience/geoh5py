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
import warnings
from typing import Any
from uuid import UUID

from geoh5py.ui_json.exceptions import UIJsonFormatError
from geoh5py.ui_json.validation import Validations

from .. import objects
from ..shared import Entity

known_types = [
    member.default_type_uid()
    for _, member in inspect.getmembers(objects)
    if hasattr(member, "default_type_uid") and member.default_type_uid() is not None
]

extra_validations = {
    "min": {"types": [int, float, type(None)]},
    "max": {"types": [int, float, type(None)]},
    "choiceList": {"types": [list, type(None)]},
    "meshType": {"types": [str, type(None)], "uuid": None},
    "dataType": {
        "values": [
            "Integer",
            "Float",
            "Text",
            "Referenced",
            "Filename",
            "Blob",
            "Vector",
            "DateTime",
            "Geometric",
            "Boolean",
        ]
    },
    "association": {"values": ["Vertex", "Cell", "Face"]},
    "parent": {"types": [str, type(None)]},
    "isValue": {"types": [bool, type(None)]},
    "property": {"types": [str, type(None)], "uuid": None},
    "dataGroupType": {
        "values": ["Multi-element", "3D vector", "Dip direction & dip", "Strike & dip"]
    },
    "filetype": {"types": [str, type(None)]},
    "fileDescription": {"types": [str, type(None)]},
    "fileMulti": {"types": [bool, type(None)]},
}


class Parameter:
    def __init__(self, name, value, validations):
        self.name: str = name
        self.value: Any = value
        self.validations: dict[str, dict] | Validations = validations

    @property
    def validations(self):
        return self._validations

    @validations.setter
    def validations(self, val):
        if isinstance(val, Validations):
            self._validations = val
        else:
            self._validations = Validations(val)

    def validate(self):
        self.validations.validate(self.name, self.value)


class FormParameter:

    form_validations = {
        "label": {"required": True, "types": [str]},
        "value": {"required": True},
        "optional": {"types": [bool, type(None)]},
        "enabled": {"types": [bool, type(None)]},
        "main": {"types": [str, type(None)]},
        "group": {"types": [str, type(None)]},
        "groupOptional": {"types": [bool, type(None)]},
        "dependency": {"types": [str, type(None)]},
        "dependencyType": {"values": ["enabled", "disabled", "show", "hide"]},
        "groupDependency": {"types": [str, type(None)]},
        "groupDependencyType": {"values": ["enabled", "disabled", "show", "hide"]},
    }
    valid_members = list(form_validations.keys())

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
        self.validate()
        form_assembly = {}
        for k in self.valid_members:
            member = getattr(self, k)
            if member.value is not None:
                form_assembly[k] = member.value

        return form_assembly

    @form.setter
    def form(self, val):

        if not all(k in val for k in ["label", "value"]):
            raise UIJsonFormatError(
                "Forms must contain both 'label' and 'value' members."
            )

        for member in self.valid_members:
            value = val.get(member, None)
            if member == "value":
                self.value = value
            elif member in self.valid_members:
                setattr(
                    self,
                    member,
                    Parameter(member, value, self.form_validations[member]),
                )
            else:
                warnings.warn(
                    f"Ignoring invalid form member {member}.  "
                    f"Valid members are: {self.valid_members}"
                )

        self.validate()

    def validate(self):
        for member in self.valid_members:
            if member != "value":
                getattr(self, member).validate()


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
            "min": {"types": [float, type(None)]},
            "max": {"types": [float, type(None)]},
        }
    form_validations = dict(
        FormParameter.form_validations,
        **integer_form_validations
    )
    valid_members = list(form_validations.keys())

    def __init__(self, name, form, validations):
        super().__init__(name, form, dict(self.base_validations, **validations))

class FloatParameter(FormParameter):

    base_validations = {"types": [float]}
    float_validations = {
            "min": {"types": [float, type(None)]},
            "max": {"types": [float, type(None)]},
            "precision": {"types": [int, type(None)]},
            "lineEdit": {"types": [bool, type(None)]}
        }
    form_validations = dict(
        FormParameter.form_validations,
        **float_validations
    )
    valid_members = list(form_validations.keys())

    def __init__(self, name, form, validations):
        super().__init__(name, form, dict(self.base_validations, **validations))

class ChoiceStringParameter(FormParameter):

    base_validations = {"types": [str]}
    choice_string_validations = {
        "choiceList": {"types": [list]}
        }
    form_validations = dict(
        FormParameter.form_validations,
        **choice_string_validations
    )
    valid_members = list(form_validations.keys())

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
        "fileDescription": {"types": [str, tuple, list]},
        "fileType": {"types": [str, tuple, list]},
        "fileMulti": {"types": [bool]},
        }
    form_validations = dict(
        FormParameter.form_validations,
        **file_validations
    )
    valid_members = list(form_validations.keys())

    def __init__(self, name, form, validations):
        super().__init__(name, form, dict(self.base_validations, **validations))
        

class UIJson:
    def __init__(self, parameters):
        self.parameters: dict[str, Parameter | FormParameter | dict[str, Any]] = parameters

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
                val[name] = parameter_class(name, value, {})
            else:
                val[name] = Parameter(name, value, {})

        self._parameters = val

    @property
    def values(self):
        return [p.value for p in self.parameters.values()]

    def validate(self):
        for parameter in self.parameters.values():
            if isinstance(parameter, Parameter):
                parameter.validate()
            else:
                parameter.value.validate()

    @staticmethod
    def identify(parameter):
        # TODO - complete this
        candidates = FormParameter.__subclasses__()
        base_members = set(FormParameter.valid_members)
        for candidate in candidates:
            identifier_members = set(candidate.valid_members).difference(base_members)
            if any(set(parameter.keys()).intersection(identifier_members)):
                return candidate
            
        return FormParameter # if no matches





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
