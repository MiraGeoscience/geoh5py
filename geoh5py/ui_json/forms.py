# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2025 Mira Geoscience Ltd.                                     '
#                                                                              '
#  This file is part of geoh5py.                                               '
#                                                                              '
#  geoh5py is free software: you can redistribute it and/or modify             '
#  it under the terms of the GNU Lesser General Public License as published by '
#  the Free Software Foundation, either version 3 of the License, or           '
#  (at your option) any later version.                                         '
#                                                                              '
#  geoh5py is distributed in the hope that it will be useful,                  '
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              '
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               '
#  GNU Lesser General Public License for more details.                         '
#                                                                              '
#  You should have received a copy of the GNU Lesser General Public License    '
#  along with geoh5py.  If not, see <https://www.gnu.org/licenses/>.           '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    PlainSerializer,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic.alias_generators import to_camel
from pydantic.functional_validators import BeforeValidator

from geoh5py.groups import Group
from geoh5py.objects import ObjectBase
from geoh5py.shared.validators import (
    empty_string_to_none,
    to_class,
    to_list,
    to_path,
    to_uuid,
    types_to_string,
    uuid_to_string,
)


class DependencyType(str, Enum):
    ENABLED = "enabled"
    DISABLED = "disabled"


class BaseForm(BaseModel):
    """
    Base class for uijson forms

    :param label: Label for ui element.
    :param value: The parameter's value.
    :param optional: If True, ui element is rendered with a checkbox to
        control the enabled state.
    :param enabled: If False, ui element is rendered grey and value will
        be written to file as None.
    :param main: Controls whether ui element will render in the general
        parameters tab (True) or optional parameters (False).
    :param tooltip: String rendered on hover over ui element.
    :param group: Grouped ui elements will be rendered within a box labelled
        with the group name.
    :param group_optional: If True, ui group is rendered with a checkbox that
        controls the enabled state of all of the groups members
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
    :param verbose: Sets the level at which Geoscience Analyst will make
        the parameter visible in a ui.json file.  Verbosity level is set
        within Analyst menu.
    """

    model_config = ConfigDict(
        extra="allow",
        frozen=True,
        populate_by_name=True,
        loc_by_alias=True,
        alias_generator=to_camel,
    )

    label: str
    value: Any
    optional: bool = False
    enabled: bool = True
    main: bool = True
    tooltip: str = ""
    group: str = ""
    group_optional: bool = False
    dependency: str = ""
    dependency_type: DependencyType = DependencyType.ENABLED
    group_dependency: str = ""
    group_dependency_type: DependencyType = DependencyType.ENABLED

    @property
    def json_string(self):
        return self.model_dump_json(exclude_unset=True, by_alias=True)

    def flatten(self):
        """Returns the data for the form."""
        return self.value

    def validate_data(self, params: dict[str, Any]):
        """Validate the form data."""


class StringForm(BaseForm):
    """
    String valued uijson form.
    """

    value: str = ""


class BoolForm(BaseForm):
    """
    Boolean valued uijson form.
    """

    value: bool = True


class IntegerForm(BaseForm):
    """
    Integer valued uijson form.
    """

    value: int = 1
    min: float = -np.inf
    max: float = np.inf


class FloatForm(BaseForm):
    """
    Float valued uijson form.
    """

    value: float = 1.0
    min: float = -np.inf
    max: float = np.inf
    precision: int = 2
    line_edit: bool = True


class ChoiceForm(BaseForm):
    """
    Choice list uijson form.
    """

    value: list[str]
    choice_list: list[str]
    multi_select: bool = False

    @field_validator("value", mode="before")
    @classmethod
    def to_list(cls, value):
        if not isinstance(value, list):
            value = [value]
        return value

    @field_serializer("value", when_used="json")
    def string_if_single(self, value):
        if len(value) == 1:
            value = value[0]
        return value

    @model_validator(mode="after")
    def valid_choice(self):
        bad_choices = []
        for val in self.value:
            if val not in self.choice_list:
                bad_choices.append(val)
        if bad_choices:
            raise ValueError(f"Provided value(s): {bad_choices} are not valid choices.")

        return self


PathList = Annotated[
    list[Path],
    BeforeValidator(to_path),
    BeforeValidator(to_list),
]


class FileForm(BaseForm):
    """
    File path uijson form
    """

    value: PathList
    file_description: list[str]
    file_type: list[str]
    file_multi: bool = False

    @field_serializer("value", when_used="json")
    def to_string(self, value):
        return ";".join([str(path) for path in value])

    @field_validator("value")
    @classmethod
    def valid_file(cls, value):
        bad_paths = []
        for path in value:
            if not path.exists():
                bad_paths.append(path)
        if any(bad_paths):
            raise ValueError(f"Provided path(s) {bad_paths} does not exist.")
        return value

    @model_validator(mode="after")
    def same_length(self):
        if len(self.file_description) != len(self.file_type):
            raise ValueError("File description and type lists must be the same length.")
        return self

    @model_validator(mode="before")
    @classmethod
    def value_file_type(cls, data):
        bad_paths = []
        for path in data["value"].split(";"):
            if Path(path).suffix[1:] not in data["file_type"]:
                bad_paths.append(path)
        if any(bad_paths):
            raise ValueError(f"Provided paths {bad_paths} have invalid extensions.")
        return data


MeshTypes = Annotated[
    list[type[ObjectBase]],
    BeforeValidator(to_class),
    BeforeValidator(to_uuid),
    BeforeValidator(to_list),
    PlainSerializer(types_to_string, when_used="json"),
]

OptionalUUID = Annotated[
    UUID | None,  # pylint: disable=unsupported-binary-operation
    BeforeValidator(empty_string_to_none),
    PlainSerializer(uuid_to_string),
]


class ObjectForm(BaseForm):
    """
    Geoh5py object uijson form.
    """

    value: OptionalUUID
    mesh_type: MeshTypes


GroupTypes = Annotated[
    list[type[Group]],
    BeforeValidator(to_class),
    BeforeValidator(to_uuid),
    BeforeValidator(to_list),
    PlainSerializer(types_to_string, when_used="json"),
]


class GroupForm(BaseForm):
    """
    Geoh5py group uijson form.
    """

    value: OptionalUUID
    group_type: GroupTypes


class Association(str, Enum):
    """
    Geoh5py object association types.
    """

    VERTEX = "Vertex"
    CELL = "Cell"
    FACE = "Face"


class DataType(str, Enum):
    """
    Geoh5py data types.
    """

    INTEGER = "Integer"
    FLOAT = "Float"
    BOOLEAN = "Boolean"
    REFERENCED = "Referenced"
    VECTOR = "Vector"
    DATETIME = "DateTime"
    GEOMETRIC = "Geometric"
    TEXT = "Text"


UUIDOrNumber = Annotated[
    UUID | float | int | None,  # pylint: disable=unsupported-binary-operation
    BeforeValidator(empty_string_to_none),
    PlainSerializer(uuid_to_string),
]


class DataForm(BaseForm):
    """
    Geoh5py data uijson form.
    """

    value: UUIDOrNumber
    parent: str
    association: Association | list[Association]
    data_type: DataType | list[DataType]
    is_value: bool = False
    property: OptionalUUID = None
    min: float = -np.inf
    max: float = np.inf
    precision: int = 2

    @model_validator(mode="after")
    def value_if_is_value(self):
        if (
            "is_value" in self.model_fields_set  # pylint: disable=unsupported-membership-test
            and self.is_value
        ):
            if isinstance(self.value, UUID):
                raise ValueError("Value must be numeric if is_value is True.")
        return self

    @model_validator(mode="after")
    def property_if_not_is_value(self):
        if (
            "is_value" in self.model_fields_set  # pylint: disable=unsupported-membership-test
            and "property" not in self.model_fields_set  # pylint: disable=unsupported-membership-test
        ):
            raise ValueError("A property must be provided if is_value is used.")
        return self

    def flatten(self):
        """Returns the data for the form."""
        if (
            "is_value" in self.model_fields_set  # pylint: disable=unsupported-membership-test
            and not self.is_value
        ):
            return self.property
        return self.value
