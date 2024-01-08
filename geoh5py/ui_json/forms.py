#  Copyright (c) 2024 Mira Geoscience Ltd.
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

from geoh5py.data import (
    BooleanData,
    DatetimeData,
    FloatData,
    IntegerData,
    ReferencedData,
    TextData,
)
from geoh5py.shared.exceptions import AggregateValidationError, BaseValidationError
from geoh5py.shared.utils import SetDict
from geoh5py.ui_json.descriptors import FormValueAccess
from geoh5py.ui_json.enforcers import EnforcerPool
from geoh5py.ui_json.parameters import (
    BoolParameter,
    FloatParameter,
    IntegerParameter,
    NumericParameter,
    Parameter,
    StringListParameter,
    StringParameter,
    TypeRestrictedParameter,
    TypeUIDRestrictedParameter,
    ValueRestrictedParameter,
)


class MemberKeys:
    """Converts in and out of camel (ui.json) and snake (python) case"""

    camel_to_snake: dict[str, str] = {
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
        "isValue": "is_value",
    }

    @property
    def snake_to_camel(self) -> dict[str, str]:
        """Gives the inverse map to camel_to_snake."""
        return {v: k for k, v in self.camel_to_snake.items()}

    def map_key(self, key: str, convention: str = "snake"):
        """Map a string from snake to camel or vice versa."""

        if convention == "snake":
            out = self.camel_to_snake.get(key, key)
        elif convention == "camel":
            out = self.snake_to_camel.get(key, key)
        else:
            raise ValueError("Convention must be 'snake' or 'camel'.")

        return out

    def map(self, collection: dict[str, Any], convention="snake"):
        """Map a dictionary from snake to camel or vice versa."""
        return {self.map_key(k, convention): v for k, v in collection.items()}


MEMBER_KEYS = MemberKeys()


class FormParameter:  # pylint: disable=too-many-instance-attributes
    """
    Base class for parameters that create visual ui elements from a form.

    :param name: Parameter name.
    :param label: Label for ui element.
    :param value: The parameter's value.
    :param enabled: If False, ui element is rendered grey and value is
        treated as None.
    :param optional: If True, ui element is rendered with a checkbox to
        control the enabled state.
    :param group_optional: If True, ui group is rendered with a checkbox
        that control the enabled state of all group members.
    :param main: Controls whether ui element will render in the general
        parameters tab (True) or optional parameters (False).
    :param group: Grouped ui elements will be rendered within a named
        box.
    :param dependency: Name of parameter that controls the enabled or
        visible state of the ui element.
    :param dependency_type: Controls whether the ui element is enabled
        or visible when the dependency is enabled if optional or True
        if a bool type.
    :param group_dependency: Name of parameter that controls the enabled
        or visible state of the ui group.
    :param group_dependency_type: Controls whether the ui group is
        enabled or visible when the group dependency is enabled if
        optional or True if a bool type.
    :param tooltip: String rendered on hover over ui element.

    :note: Standardized form members are accessible through public namespace
        by way of the FormValueAccess descriptor.
    """

    static_validations = {"required_form_members": ["label", "value"]}
    identifier_members: list[str] = []

    def __init__(
        self,
        name: str,
        value: Any | None = None,
        **kwargs,
    ):
        self.name: str = name
        self._value: Parameter = self._set_value_parameter(value)
        self._label = StringParameter("label")
        self._enabled = BoolParameter("enabled", value=True)
        self._optional = BoolParameter("optional")
        self._group_optional = BoolParameter("group_optional")
        self._main = BoolParameter("main", value=True)
        self._group = StringParameter("group")
        self._dependency = StringParameter("dependency")
        self._dependency_type = ValueRestrictedParameter(
            "dependency_type", ["enabled", "disabled"], value="enabled"
        )
        self._group_dependency = StringParameter("group_dependency")
        self._group_dependency_type = ValueRestrictedParameter(
            "group_dependency_type", ["enabled", "disabled"], value="enabled"
        )
        self._tooltip = StringParameter("tooltip")
        self._allow_values_access()
        self._extra_members: dict[str, Any] = {}
        self._active_members: list[str] = []
        if kwargs:
            self.register(kwargs)
        self._validations = SetDict()
        self.enforcers: EnforcerPool = EnforcerPool.from_validations(
            self.name, self.validations
        )

    @property
    def validations(self):
        """Returns a dictionary of static and inferred validations."""
        if not self._validations:
            self._validations.update(self.dynamic_validations)
            self._validations.update(self.static_validations)

        return self._validations

    @property
    def dynamic_validations(self):
        """Infer validations from parameters."""
        validations = SetDict()
        if "group_optional" in self.active:
            validations.update({"required": "group"})

        return validations

    @property
    def uijson_validations(self):
        """Validations for UIJson level enforcers."""
        validations = SetDict()
        if "dependency" in self.active:
            validations.update({"required": self.dependency})  # type: ignore

        return validations

    def form(self, use_camel: bool = False) -> dict[str, Any]:
        """
        Returns dictionary of active form members and their values.

        :param use_camel: If True, keys are converted to camel case.
        """
        form = {}
        for member in self.active:
            if member in self._extra_members:
                form[member] = self._extra_members[member]
            else:
                form[member] = getattr(self, f"_{member}").value

        if use_camel:
            form = MEMBER_KEYS.map(form, "camel")

        return form

    def register(self, members: dict[str, Any]):
        """
        Update form members with incoming values.

        :param members: Dictionary of form members and associated data.
        """

        error_list = []
        members = MEMBER_KEYS.map(members)
        for member in list(members):
            if member in self.valid_members:
                try:
                    setattr(self, member, members.pop(member))
                except BaseValidationError as err:
                    error_list.append(err)

        if error_list:
            if len(error_list) == 1:
                raise error_list.pop()
            raise AggregateValidationError(self.name, error_list)

        self._extra_members.update(members)

    def validate(self):
        """Validates form data against the pool of enforcers."""
        self.enforcers.enforce(self.form())

    @property
    def valid_members(self) -> list[str]:
        """Recognized form member names."""
        exclusions = ["_extra_members", "_active_members", "_validations"]
        private_attrs = [k for k in self.__dict__ if k.startswith("_")]
        return [k[1:] for k in private_attrs if k not in exclusions]

    @property
    def active(self) -> list[str]:
        """
        Returns names of active form members.

        :return: List of active form members.  These will include any members
            that were:
                1. Provided during construction.
                2. Updated through the 'register' method.
                3. Updated through member setters.
        """
        active = ["value"] + self._active_members + list(self._extra_members)
        return list(dict.fromkeys(active))  # Unique while preserving order

    @classmethod
    def is_form(cls, form: dict[str, Any]) -> bool:
        """Returns True if form contains any identifier members."""
        id_members = cls.identifier_members
        form_members = MEMBER_KEYS.map(form)
        return any(k in form_members for k in id_members)

    @property
    def value(self):
        return self._value.value

    @value.setter
    def value(self, val):
        self._value.value = val

    def _set_value_parameter(self, value) -> Parameter:
        """Handles value argument as either a Parameter or a value."""

        if isinstance(value, Parameter):
            value.name = "value"
            out = value
        else:
            out = Parameter("value", value)

        return out

    def _allow_values_access(self):
        """Valid members public attr accesses underlying parameter value."""
        for member in self.valid_members:
            if member not in dir(self):
                setattr(self.__class__, member, FormValueAccess(f"_{member}"))

    def __str__(self):
        return f"<{type(self).__name__}> : '{self.name}' -> {self.value}"

    def __contains__(self, item):
        return MEMBER_KEYS.map_key(item) in self.active


class StringFormParameter(FormParameter):
    """String parameter type."""

    def __init__(self, name, value=None, **kwargs):
        value = StringParameter("value", value=value)
        super().__init__(name, value=value, **kwargs)


class BoolFormParameter(FormParameter):
    """Boolean parameter type."""

    def __init__(self, name, value: bool = False, **kwargs):
        param = BoolParameter("value", value=value)
        super().__init__(name, value=param, **kwargs)


class IntegerFormParameter(FormParameter):
    """
    Integer parameter type.

    :param min: Minimum value for ui element.
    :param max: Maximum value for ui element.
    """

    identifier_members: list[str] = []

    def __init__(self, name, value=None, **kwargs):
        self._min = IntegerParameter("min")
        self._max = IntegerParameter("max")
        value = IntegerParameter("value", value=value)
        super().__init__(name, value=value, **kwargs)


class FloatFormParameter(FormParameter):
    """
    Float parameter type.

    :param min: Minimum value for ui element.
    :param max: Maximum value for ui element.
    :param precision: Number of decimal places to display in ui element.
    :param line_edit: If False, the ui element incluces a spinbox.
    """

    identifier_members: list[str] = ["precision", "line_edit"]

    def __init__(self, name, value=None, **kwargs):
        self._min = FloatParameter("min")
        self._max = FloatParameter("max")
        self._precision = IntegerParameter("precision")
        self._line_edit = BoolParameter("line_edit", value=True)
        value = FloatParameter("value", value=value)
        super().__init__(name, value=value, **kwargs)


class ChoiceStringFormParameter(FormParameter):
    """
    Choice string parameter type.

    :param choice_list: List of choices for ui dropdown.
    """

    identifier_members: list[str] = ["choice_list"]
    static_validations = {"required_form_members": ["choice_list"]}

    def __init__(self, name, choice_list, value=None, **kwargs):
        self._choice_list = StringListParameter("choice_list")
        value = ValueRestrictedParameter("value", choice_list, value=value)
        super().__init__(name, value=value, choice_list=choice_list, **kwargs)


class FileFormParameter(FormParameter):
    """
    File parameter type.

    :param file_description: list of file descriptions for each file type.
    :param file_type: list of file extensions to filter directory on.
    :param file_multi: Allow multiple files to be selected from dropdown.
    """

    identifier_members: list[str] = ["file_description", "file_type", "file_multi"]
    static_validations = {"required_form_members": ["file_description", "file_type"]}

    def __init__(self, name, value=None, **kwargs):
        self._file_description = StringListParameter("file_description")
        self._file_type = StringListParameter("file_type")
        self._file_multi = BoolParameter("file_multi")
        value = StringParameter("value", value=value)
        super().__init__(name, value=value, **kwargs)


class ObjectFormParameter(FormParameter):
    """
    Object parameter type.

    :param mesh_type: list of object types (uid) that will be available in the
        dropdown.  Empty list will reveal all objects in geoh5.
    """

    identifier_members: list[str] = ["mesh_type"]
    static_validations = dict(
        FormParameter.static_validations, **{"required_form_members": ["mesh_type"]}
    )

    def __init__(self, name, mesh_type, value=None, **kwargs):
        self._mesh_type = StringListParameter("mesh_type", value=[])
        value = TypeUIDRestrictedParameter("value", mesh_type, value=value)
        super().__init__(name, value=value, **kwargs)

    @property
    def uijson_validations(self):
        """Validations for UIJson level enforcers."""
        validations = SetDict()
        if self.value is not None:
            validations.update({"required_workspace_object": self.name})

        return validations


DATA_TYPES = {
    "Integer": IntegerData,
    "Float": FloatData,
    "Text": TextData,
    "Referenced": ReferencedData,
    "DateTime": DatetimeData,
    "Boolean": BooleanData,
}


class DataFormParameter(FormParameter):
    """
    Data parameter type.

    :param parent: Name of parent object.
    :param association: Filters data to those living on vertices or cells.
    :param data_type: Filters data type.
    :param data_group_type: Filters data group type.
    """

    identifier_members: list[str] = ["data_group_type"]
    static_validations = {
        "required_form_members": ["parent", "association", "data_type"]
    }

    def __init__(self, name, data_type, value=None, **kwargs):
        self._parent = StringParameter("parent")
        self._association = ValueRestrictedParameter(
            "association", ["Vertex", "Cell", "Face"]
        )
        self._data_type = ValueRestrictedParameter("data_type", list(DATA_TYPES))
        self._data_group_type = ValueRestrictedParameter(
            "data_group_type", ["3D vector", "Dip direction & dip", "Strike & dip"]
        )
        value = TypeRestrictedParameter(
            "value", [DATA_TYPES.get(data_type, None)], value=value
        )
        super().__init__(name, value=value, data_type=data_type, **kwargs)

    @property
    def uijson_validations(self):
        """Validations for UIJson level enforcers."""

        validations = SetDict()
        if self.value is not None:
            validations.update(
                {
                    "required_workspace_object": self.name,
                    "required_object_data": [(self._parent.value, self.name)],
                }
            )

        return validations


class DataValueFormParameter(FormParameter):
    """
    Data value parameter type.

    :param parent: Name of parent object.
    :param association: Filters data to those living on vertices or cells.
    :param data_type: Filters data type.
    :param is_value: Gives ui element a button to switch between value box
        and dropdown of available properties.
    :param property: Name of property.
    """

    identifier_members: list[str] = ["is_value", "property"]
    static_validations = {
        "required_form_members": [
            "parent",
            "association",
            "data_type",
            "is_value",
            "property",
        ]
    }

    def __init__(self, name, data_type, value=None, **kwargs):
        self._parent = StringParameter("parent")
        self._association = ValueRestrictedParameter(
            "association", ["Vertex", "Cell", "Face"]
        )
        self._data_type = ValueRestrictedParameter("data_type", list(DATA_TYPES))
        self._is_value = BoolParameter("is_value")
        self._property = TypeRestrictedParameter(
            "property", [DATA_TYPES.get(data_type, type(None))]
        )
        value = NumericParameter("value", value=value)
        super().__init__(name, value=value, data_type=data_type, **kwargs)

    @property
    def value(self):
        """Form value is value of property when is_value is False."""
        val = self.property  # type: ignore # pylint: disable=no-member
        if self.is_value:  # type: ignore # pylint: disable=no-member
            val = self._value.value
        return val

    @value.setter
    def value(self, val):
        if isinstance(val, (int, float)):
            self._value.value = val
        else:
            self._property.value = val
