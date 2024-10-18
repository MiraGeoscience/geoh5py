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

# pylint: disable=R0913

from __future__ import annotations

import inspect
import uuid
from uuid import UUID

from .. import groups, objects
from ..shared import Entity


known_object_types = [
    member.default_type_uid()
    for _, member in inspect.getmembers(objects)
    if hasattr(member, "_TYPE_UID") and member.default_type_uid() is not None
]

known_group_types = [
    member.default_type_uid()
    for _, member in inspect.getmembers(groups)
    if hasattr(member, "_TYPE_UID") and member.default_type_uid() is not None
]


def optional_parameter(state: str) -> dict[str, bool]:
    """
    Returns dictionary to make existing ui optional via .update() method.

    :param state: Initial state of optional parameter. Can be 'enabled' or 'disabled'.
    """

    form_opt = {"optional": True}
    if state == "enabled":
        form_opt["enabled"] = True
    elif state == "disabled":
        form_opt["enabled"] = False
    else:
        raise ValueError(
            "Unrecognized state option. Must be either 'enabled' or 'disabled'."
        )

    return form_opt


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
    *,
    main: bool = True,
    label: str = "Integer data",
    value: int = 1,
    vmin: int | None = None,
    vmax: int | None = None,
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
    form = {"main": main, "label": label, "value": value}

    for prop, val in {"vmin": vmin, "vmax": vmax}.items():
        if val is not None:
            form[prop] = val

    if optional is not None:
        form.update(optional_parameter(optional))
    return form


def float_parameter(
    *,
    main: bool = True,
    label: str = "Float data",
    value: float = 1.0,
    vmin: float | None = None,
    vmax: float | None = None,
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
        "precision": precision,
        "lineEdit": line_edit,
    }

    for prop, val in {"vmin": vmin, "vmax": vmax}.items():
        if val is not None:
            form[prop] = val

    if optional is not None:
        form.update(optional_parameter(optional))

    return form


def string_parameter(
    *,
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
    *,
    choice_list: tuple = ("Option A", "Option B"),
    label: str = "String data",
    main: bool = True,
    multi_select: bool = False,
    optional: str | None = None,
    value: str = "Option A",
) -> dict:
    """
    Dropdown menu of string choices.

    :param main: Show form in main.
    :param choice_list: List of options.
    :param label: Label identifier.
    :param multi_select: Option to select multiple choices.
    :param value: Input value.
    :param optional: Make optional if not None. Initial state provided by not None
        value.  Can be either 'enabled' or 'disabled'.

    :returns: Ui_json compliant dictionary.
    """
    form = {
        "main": main,
        "multiSelect": multi_select,
        "label": label,
        "value": value,
        "choiceList": choice_list,
    }

    if optional is not None:
        form.update(optional_parameter(optional))
    return form


def file_parameter(
    *,
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


def group_parameter(
    *,
    main: bool = True,
    label: str = "Object",
    group_type: tuple = tuple(known_group_types),
    value: str | None = None,
    optional: str | None = None,
) -> dict:
    """
    Dropdown menu of groups of specific types.

    :param main: Show form in main.
    :param label: Label identifier.
    :param value: Input value.
    :param group_type: Type of selectable groups.
    :param optional: Make optional if not None. Initial state provided by not None
        value.  Can be either 'enabled' or 'disabled'.

    :returns: Ui_json compliant dictionary.
    """
    form = {"main": main, "label": label, "value": value, "groupType": group_type}

    if optional is not None:
        form.update(optional_parameter(optional))
    return form


def object_parameter(
    *,
    main: bool = True,
    label: str = "Object",
    mesh_type: tuple = tuple(known_object_types),
    multi_select: bool = False,
    value: str | None = None,
    optional: str | None = None,
) -> dict:
    """
    Dropdown menu of objects of specific types.

    :param main: Show form in main.
    :param label: Label identifier.
    :param value: Input value.
    :param multi_select: Option to select multiple choices.
    :param mesh_type: Type of selectable objects.
    :param optional: Make optional if not None. Initial state provided by not None
        value.  Can be either 'enabled' or 'disabled'.

    :returns: Ui_json compliant dictionary.
    """
    form = {
        "main": main,
        "label": label,
        "value": value,
        "multiSelect": multi_select,
        "meshType": mesh_type,
    }

    if optional is not None:
        form.update(optional_parameter(optional))
    return form


def data_parameter(
    *,
    main: bool = True,
    label: str = "Data channel",
    association: str = "Vertex",
    data_type: str = "Float",
    data_group_type: str | None = None,
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
    :param multi_select: Option to select multiple choices.
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
    *,
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


def drillhole_group_data(
    *,
    value: list[str] | None = None,
    main: bool = True,
    label: str = "Data channel",
    group_type: UUID | None = groups.DrillholeGroup.default_type_uid(),
    group_value: UUID | None = None,
    multiselect: bool = True,
    optional: str | None = None,
    enabled: bool = True,
    tooltip: str = "Select the data channel to use for the calculation.",
) -> dict:
    """
    Dropdown of data or input box.

    :param main: Show form in main.
    :param label: Label identifier.
    :param value: Input value.
    :param group_type: The group type to select, must be drillhole group.
    :param group_value: The selected group UUID.
    :param multiselect: Allow multiple data selection.
    :param value: The name of the Data channel to extract.
    :param optional: Make optional if not None. Initial state provided by not None
    :param enabled: Enable or disable the form.
    :param tooltip: The tooltip to display when hovering over the form.

    :returns: Ui_json compliant dictionary.
    """
    form = {
        "main": main,
        "label": label,
        "groupType": group_type,
        "groupValue": group_value,
        "multiselect": multiselect,
        "value": value,
        "optional": optional,
        "enabled": enabled,
        "tooltip": tooltip,
    }

    if optional is not None:
        form.update(optional_parameter(optional))
    return form


def range_label_template(
    *,
    main: bool = True,
    label: str = "Range",
    allow_complement: bool = False,
    is_complement: bool = False,
    parent: str = "",
    property_: str | uuid.UUID = "",
    association: str = "Vertex",
    data_type: list[str] | str = "Float",
    value: list[float | int] | None = None,
    range_label: str = "Values",
    enabled: bool = True,
    optional: str | None = None,
) -> dict:
    """
    Template for range label.

    Or it is a multiselect data on a property group if referenced data.
    Or they are 2 values, an upper and lower bound as float values.
    In the second scenario, is_complement allows to invert the range.

    :param main: Show form in main.
    :param label: The label identifier.
    :param allow_complement: If users can invert the range.
    :param is_complement: If the range is inverted.
    :param parent: The parent object.
    :param property_: The property uid associated with the range.
    :param association: The association type.
    :param data_type: The data type of the range.
    :param value: The value of the range.
    :param range_label: The label of the range.
    :param enabled: The state of the form.
    :param optional: Make optional if not None. Initial state provided by not None.

    :return: The form dictionary.
    """
    form = {
        "main": main,
        "label": label,
        "allowComplement": allow_complement,
        "isComplement": is_complement,
        "parent": parent,
        "property": property_,
        "association": association,
        "dataType": data_type,
        "value": value,
        "rangeLabel": range_label,
        "enabled": enabled,
    }
    if optional is not None:
        form.update(optional_parameter(optional))

    return form
