# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2020-2026 Mira Geoscience Ltd.                                '
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

from abc import ABC, abstractmethod
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

from geoh5py.data import DataAssociationEnum, DataTypeEnum
from geoh5py.groups import Group, GroupTypeEnum
from geoh5py.objects import ObjectBase
from geoh5py.shared.validators import (
    to_class,
    to_list,
    to_path,
    to_uuid,
    types_to_string,
)
from geoh5py.ui_json.annotations import OptionalUUIDList, OptionalValueList
from geoh5py.ui_json.validations.form import (
    empty_string_to_none,
    uuid_to_string,
    uuid_to_string_or_numeric,
)


# pylint: disable=too-many-return-statements, too-many-branches


class DependencyType(str, Enum):
    ENABLED = "enabled"
    DISABLED = "disabled"
    SHOW = "show"
    HIDE = "hide"


class BaseForm(ABC, BaseModel):
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
    :param placeholder_text: Text displayed in ui element when no data
        has been provided.
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
    placeholder_text: str = ""

    @classmethod
    def infer(cls, data: dict[str, Any]) -> type[BaseForm]:
        """
        Infer and return the appropriate form.

        :param data: Dictionary of form data.
        """
        data = {to_camel(k): v for k, v in data.items()}
        if "choiceList" in data:
            if data.get("multiSelect", False):
                return MultiChoiceForm
            return ChoiceForm
        if "originalLabel" in data and "alternateLabel" in data:
            return RadioLabelForm
        if any(k in data for k in ["fileDescription", "fileType"]):
            return FileForm
        if "meshType" in data:
            return ObjectForm
        if "groupType" in data:
            return GroupForm
        if any(
            k in data
            for k in ["parent", "association", "dataType", "isValue", "property"]
        ):
            if any(
                k in data for k in ["allowComplement", "isComplement", "rangeLabel"]
            ):
                return DataRangeForm
            if "dataGroupType" in data:
                return DataGroupForm
            if "multiSelect" in data:
                return MultiSelectDataForm
            if any(
                k in data for k in ["min", "max", "precision", "isValue", "property"]
            ):
                return DataOrValueForm
            return DataForm
        if isinstance(data.get("value"), str):
            return StringForm
        if isinstance(data.get("value"), bool):
            return BoolForm
        if isinstance(data.get("value"), int):
            return IntegerForm
        if isinstance(data.get("value"), float):
            return FloatForm

        raise ValueError(f"Could not infer form from data: {data}")

    @abstractmethod
    def type_check(self, form: dict[str, Any]) -> bool:
        pass

    @property
    def json_string(self) -> str:
        """Returns the form as a json string."""
        return self.model_dump_json(exclude_unset=True, by_alias=True)

    def flatten(self):
        """Returns the data for the form."""
        return self.value

    def validate_data(self, params: dict[str, Any]):
        """Validate the form data."""


class StringForm(BaseForm):
    """
    String valued uijson form.

    Shares documented attributes with the BaseForm.
    """

    value: str = ""

    def type_check(self, form: dict[str, Any]):
        check = False
        if isinstance(form.get("value", None), str):
            is_choice_form = "choice_list" in form
            is_radio_label_form = "original_label" in form or "alternate_label" in form
            if not is_choice_form or not is_radio_label_form:
                check = True
        return check


class RadioLabelForm(StringForm):
    """
    Radio button for two-option strings.

    Shares documented attributes with the BaseForm.

    The uijson dialogue will render two radio buttons with label choices.  Any
    form labels within the ui.json containing the string matching the original
    button will be altered to the reflect the new button choice.

    :param original_label: Label for the original value.
    :param alternative_label: Label for the alternative value.
    """

    original_label: str = ""
    alternate_label: str = ""


class BoolForm(BaseForm):
    """
    Boolean valued uijson form.

    Shares documented attributes with the BaseForm.
    """

    value: bool = True


class IntegerForm(BaseForm):
    """
    Integer valued uijson form.

    Shares documented attributes with the BaseForm.

    :param min: Minimum value accepted by the rendered form in
        Geoscience ANALYST.
    :param max: Maximum value accepted by the rendered form in
        Geoscience ANALYST.
    """

    value: int = 1
    min: float = -np.inf
    max: float = np.inf


class FloatForm(BaseForm):
    """
    Float valued uijson form.

    Shares documented attributes with the BaseForm.

    :param min: Minimum value accepted by the rendered form in
        Geoscience ANALYST.
    :param max: Maximum value accepted by the rendered form in
        Geoscience ANALYST.
    :param precision: Number of decimal places rendered in Geoscience
        ANALYST.
    :param line_edit: If True, Geoscience ANALYST will render a spin box
        for adjusting the value by an increment controlled by the precision.
    """

    value: float = 1.0
    min: float = -np.inf
    max: float = np.inf
    precision: int = 2
    line_edit: bool = True


class ChoiceForm(BaseForm):
    """
    Choice list uijson form.

    Shares documented attributes with the BaseForm.

    :param choice_list: List of valid choices for the form.  The choices
        are rendered in Geoscience ANALYST as a dropdown menu.

    """

    value: str = ""
    choice_list: list[str]

    @model_validator(mode="after")
    def valid_choice(self):
        if self.value not in self.choice_list:
            raise ValueError(f"Provided value: '{self.value}' is not a valid choice.")

        return self


class MultiChoiceForm(BaseForm):
    """
    Multi-choice list uijson form.

    Shares documented attributes with the BaseForm.

    :param choice_list: List of valid choices for the form.  The choices
        are rendered in Geoscience ANALYST as a multi-selection dropdown
        menu.
    :param multi_select: Must be True for MultiChoiceForm.
    """

    value: list[str]
    choice_list: list[str]
    multi_select: bool = True

    @field_validator("multi_select", mode="before")
    @classmethod
    def only_multi_select(cls, value):
        if not value:
            raise ValueError("MultiChoiceForm must have multi_select: True.")
        return value

    @field_validator("value", mode="before")
    @classmethod
    def to_list(cls, value: str | list[str]) -> list[str]:
        if not isinstance(value, list):
            value = [value]
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
    File path uijson form.

    Shares documented attributes with the BaseForm.

    :param file_description: List of file descriptions for each file type.
    :param file_type: List of file extensions (without the dot) for each file type.
    :param file_multi: If True, Geoscience ANALYST will allow multi-selection of
        files.
    :param directory_only: If True, Geoscience ANALYST will restrict selecitons
        to directories only.
    """

    value: PathList
    file_description: list[str]
    file_type: list[str]
    file_multi: bool = False
    directory_only: bool = False

    @field_serializer("value", when_used="json")
    def to_string(self, value: list[Path]) -> str:
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

    @model_validator(mode="after")
    def value_file_type(self):
        bad_paths = []
        for path in self.value:
            if not self.directory_only and Path(path).suffix[1:] not in self.file_type:
                bad_paths.append(path)
        if any(bad_paths):
            raise ValueError(f"Provided paths {bad_paths} have invalid extensions.")
        return self

    @model_validator(mode="before")
    @classmethod
    def directory_file_type(cls, data):
        if data.get("directoryOnly", False) and data["fileType"] != ["directory"]:
            raise ValueError(
                "File type must be ['directory'] if directory_only is True."
            )
        if data.get("directoryOnly", False) and data["fileDescription"] != [
            "Directory"
        ]:
            raise ValueError(
                "File description must be ['Directory'] if directory_only is True."
            )
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

    Shares documented attributes with the BaseForm.

    :param mesh_type: List of object types that restricts the options in the
        Geoscience ANALYST ui.json dropdown.
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

    Shares documented attributes with the BaseForm.

    :param group_type: List of group types that restricts the options in the
        Geoscience ANALYST ui.json dropdown.
    """

    value: OptionalUUID
    group_type: GroupTypes


Association = Enum(  # type: ignore
    "Association",
    [(k.name, k.name.capitalize()) for k in DataAssociationEnum],
    type=str,
)

DataType = Enum(  # type: ignore
    "DataType", [(k.name, k.name.capitalize()) for k in DataTypeEnum], type=str
)

UUIDOrNumber = Annotated[
    UUID | float | int | None,  # pylint: disable=unsupported-binary-operation
    BeforeValidator(empty_string_to_none),
    PlainSerializer(uuid_to_string_or_numeric),
]


class DataFormMixin(BaseModel):
    """
    Mixin class to add common attributes a series of data classes.

    Shares documented attributes with the BaseForm.

    :param parent: The name of the parameter in the ui.json that contains
        the data to select from.
    :param association: The data association, eg: 'Cell', 'Face', 'Vertex'
        of a grid object, that filters the options in the Geoscience ANALYST
        ui.json dropdown.
    :param data_type: The data type, eg: 'Integer', 'Float', that filters
        the options in the Geoscience ANALYST ui.json dropdown.
    """

    parent: str
    association: Association | list[Association]
    data_type: DataType | list[DataType]


class DataForm(DataFormMixin, BaseForm):
    """
    Geoh5py uijson form for data associated with an object.

    Shares documented attributes with the BaseForm and DataFormMixin.
    """

    value: OptionalUUID


class DataGroupForm(DataForm):
    """
    Geoh5py uijson form for grouped data associated with an object.

    Shares documented attributes with the BaseForm.

    :param data_group_type: The group type, eg: 'Multi-Element', '3d Vector'
        that filters the groups available in the Geoscience ANALYST ui.json.
    """

    data_group_type: GroupTypeEnum | list[GroupTypeEnum]


class DataOrValueForm(DataFormMixin, BaseForm):
    """
    Geoh5py uijson data form that also accepts a single value.

    Shares documented attributes with the BaseForm and DataFormMixin.

    :param is_value: If True, the value field is used to provide a scalar value.
    """

    value: UUIDOrNumber
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

    def flatten(self) -> UUID | float | int | None:
        """Returns the data for the form."""
        if (
            "is_value" in self.model_fields_set  # pylint: disable=unsupported-membership-test
            and not self.is_value
        ):
            return self.property
        return self.value


class MultiSelectDataForm(DataFormMixin, BaseForm):
    """
    Geoh5py uijson data form with multi-selection.

    Shares documented attributes with the BaseForm and DataFormMixin.

    :param multi_select: Must be True for MultiSelectDataForm.
    """

    value: OptionalUUIDList
    multi_select: bool = True

    @field_validator("multi_select", mode="before")
    @classmethod
    def only_multi_select(cls, value: bool) -> bool:
        """Validate that multi_select is True."""
        if not value:
            raise ValueError("MultiSelectForm must have multi_select: True.")
        return value

    @field_validator("value", mode="before")
    @classmethod
    def to_list(cls, value):
        if not isinstance(value, list):
            value = [value]
        return value


class DataRangeForm(DataFormMixin, BaseForm):
    """
    Geoh5py data range uijson form.

    Shares documented attributes with the BaseForm and the DataFormMixin.

    :param value: The value can be a single float or a list of two floats.
        Geoscience ANALYST will estimate a range on load if a single float
        is provided, but will always return a list.
    :param property: The UUID of the property to which the range applies.
    :param range_label: Label for the range.
    :param allow_complement: If True, the complement option will be available
        in Geoscience ANALYST as a checkbox.
    :param is_complement: If True, the range slider in Geoscience ANALYST will
        be inverted and the implied selection is outside of the range provided.
    """

    value: OptionalValueList
    property: OptionalUUID
    range_label: str
    allow_complement: bool = False
    is_complement: bool = False
