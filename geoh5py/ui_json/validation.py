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

from collections.abc import Callable
from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID
from warnings import warn

from geoh5py import Workspace
from geoh5py.groups import PropertyGroup
from geoh5py.objects import ObjectBase
from geoh5py.shared import Entity
from geoh5py.shared.exceptions import RequiredValidationError
from geoh5py.shared.validators import (
    AssociationValidator,
    AtLeastOneValidator,
    BaseValidator,
    OptionalValidator,
    PropertyGroupValidator,
    RequiredValidator,
    ShapeValidator,
    TypeValidator,
    UUIDValidator,
    ValueValidator,
)
from geoh5py.ui_json.forms import BoolForm, DataOrValueForm
from geoh5py.ui_json.utils import requires_value


if TYPE_CHECKING:
    from geoh5py.ui_json.ui_json import BaseUIJson

Validation = dict[str, Any]


class InputValidation:
    """
    Validations on dictionary of parameters.

    Attributes
    ----------
    validations : Validations dictionary with matching set of input parameter keys.
    ignore_requirements (optional): Omit raising error on 'required' validator.

    Methods
    -------
    validate_data(data)
        Validates data of params and contents/type/shape/keys/reqs of values.
    """

    def __init__(
        self,
        validators: dict[str, BaseValidator] | None = None,
        validations: dict[str, Any] | None = None,
        ui_json: dict[str, Any] | None = None,
        validation_options: dict[str, Any] | None = None,
    ):
        self.validations = self.infer_validations(ui_json, validations=validations)
        self.validators: dict[str, BaseValidator] = validators

        if validation_options is None:
            validation_options = {}

        self.ignore_list: tuple = validation_options.get("ignore_list", ())
        self.ignore_requirements: bool = validation_options.get(
            "ignore_requirements", False
        )

    @property
    def validations(self):
        return self._validations

    @validations.setter
    def validations(self, val: dict[str, Any]):
        self._validations = val

    @property
    def validators(self):
        return self._validators

    @validators.setter
    def validators(self, val: dict[str, BaseValidator]):
        if val is None:
            val = {}
        elif not all(isinstance(v, BaseValidator) for v in val.values()):
            raise TypeError("Validators should be subclass of BaseValidator.")

        if self.validations is not None:
            required_validators = InputValidation._required_validators(self.validations)
            val = dict(required_validators, **val)

        self._validators = val

    @staticmethod
    def _unique_validators(validations: dict[str, Any]) -> list[str]:
        """Return names of validators required by a validations dictionary."""
        return list({key for item in validations.values() for key in item})

    @staticmethod
    def _required_validators(validations):
        """Returns dictionary of validators required by validations."""
        unique_validators = InputValidation._unique_validators(validations)
        sub_classes: list[type[BaseValidator]] = BaseValidator.__subclasses__()
        all_validators: dict[str, Any] = {k.validator_type: k() for k in sub_classes}
        val = {}
        for k in unique_validators:
            if k not in all_validators:
                raise ValueError(f"No validator implemented for argument '{k}'.")
            val[k] = all_validators[k]
        return val

    @staticmethod
    def _validations_from_uijson(  # pylint: disable=too-many-branches
        ui_json: dict[str, Any],
    ) -> dict[str, dict]:
        """Determine base set of validations from ui.json structure."""
        validations: dict[str, dict] = {}
        for key, item in ui_json.items():
            if not isinstance(item, dict):
                check_type = cast(Any, type(item))
                validations[key] = {
                    "types": [check_type],
                }
                continue

            if "isValue" in item:
                validations[key] = {
                    "types": [str, UUID, int, float, Entity],
                }
                try:
                    validations[key]["association"] = item["parent"]
                    validations[key]["uuid"] = None
                except KeyError as error:
                    warn(
                        f"Failed to set association for {key}. {error}",
                    )
            elif "rangeLabel" in item and "isComplement" not in item:
                validations[key] = {
                    "types": [list],
                }
            elif ("groupValue" in item or "isComplement" in item) and "value" in item:
                validations[key] = {
                    "types": [dict],
                }
            elif "choiceList" in item:
                validations[key] = {
                    "types": [str],
                    "values": item["choiceList"],
                }
            elif "fileType" in item:
                validations[key] = {
                    "types": [str],
                }
            elif "meshType" in item or "groupType" in item:
                validations[key] = {
                    "types": [str, UUID, Entity],
                    "association": "geoh5",
                    "uuid": None,
                }
            elif "parent" in item:
                validations[key] = {
                    "types": [str, UUID, Entity],
                    "association": item["parent"],
                    "uuid": None,
                }
                if "dataGroupType" in item:
                    validations[key]["property_group_type"] = item["dataGroupType"]
                    validations[key]["types"] = [str, UUID, PropertyGroup]
            elif "value" in item:
                check_type = str
                if item["value"] is not None:
                    check_type = cast(Any, type(item["value"]))

                validations[key] = {
                    "types": (
                        [check_type, Entity] if check_type is UUID else [check_type]
                    ),
                }

            validations[key].update({"optional": not requires_value(ui_json, key)})

            if "types" in validations[key]:
                if not requires_value(ui_json, key):
                    validations[key]["types"] += [type(None)]

                if item.get("multiSelect", False):
                    validations[key]["types"] += [list]

        return validations

    @staticmethod
    def infer_validations(
        ui_json: dict[str, Any] | None, validations: dict[str, dict] | None = None
    ) -> dict:
        """Infer necessary validations from ui json structure."""

        inferred_validations = (
            {} if ui_json is None else InputValidation._validations_from_uijson(ui_json)
        )

        if validations is not None:
            inferred_validations = InputValidation._merge_validations(
                inferred_validations, validations
            )

        return inferred_validations

    @staticmethod
    def _merge_validations(
        validations_a: dict[str, dict], validations_b: dict[str, dict]
    ):
        """Merge validations_b into validations_a."""
        out = deepcopy(validations_a)
        if validations_b is not None:
            for key, val in validations_b.items():
                if key not in out:
                    out[key] = val
                else:
                    out[key].update(val)
        return out

    def validate(
        self, name: str, value: Any, validations: dict[str, Any] | None = None
    ):
        """
        Run validations on a given key and value.

        :param name: Parameter identifier.
        :param value: Input parameter value.
        :param validations: [Optional] Validations provided on runtime
        """

        if validations is None:
            if name not in self.validations:
                raise KeyError(f"{name} is missing from the known validations.")

            validations = self.validations[name]

        for validator in [
            RequiredValidator,
            AtLeastOneValidator,
            OptionalValidator,
            TypeValidator,
            UUIDValidator,
            AssociationValidator,
            PropertyGroupValidator,
            ValueValidator,
            ShapeValidator,
        ]:
            val = str(validator.validator_type)
            if (
                val not in validations
                or (val == "required" and self.ignore_requirements)
                or name in self.ignore_list
            ):
                continue
            self.validators[val](name, value, validations[val])

    def validate_data(self, data: dict[str, Any]) -> None:
        """
        Calls validate method on ui.json data structure for cross-dependencies.

        Individual key, value pairs are validated for expected type.

        :param data: Input data with known validations.
        """

        one_of_validations: dict[str, Any] = {}
        local_validations = self.validations.copy()
        for param, validations in local_validations.items():
            if param not in data.keys():
                if "required" in validations and not self.ignore_requirements:
                    raise RequiredValidationError(param)

                continue

            if "one_of" in validations:
                one_of_group = validations.pop("one_of")
                val = {param: data[param] is not None}
                if one_of_group in one_of_validations:
                    one_of_validations[one_of_group].update(val)
                else:
                    one_of_validations[one_of_group] = val

            if "association" in validations and validations["association"] in data:
                valid = validations.copy()
                valid["association"] = data[validations["association"]]
                self.validate(param, data[param], valid)
            else:
                self.validate(param, data.get(param, None), validations)

        for name, val in one_of_validations.items():
            self.validate(name, val, {"one_of": None})

    def __call__(self, data, *args):
        if isinstance(data, dict):
            self.validate_data(data)
        elif isinstance(data, str) and args is not None:
            self.validate(data, args)
        else:
            raise ValueError(
                "InputValidators can only be called with dictionary of data or "
                "(key, value) pair."
            )


## Validation utility for UIJson class ##
class UIJsonError(Exception):
    """Exception raised for errors in the UIJson object."""

    def __init__(self, message: str):
        super().__init__(message)


class ErrorPool:  # pylint: disable=too-few-public-methods
    """
    Stores validation errors for all UIJson members.

    :param errors: Dictionary mapping parameter names to lists of
        exceptions encountered during validation.
    """

    def __init__(self, errors: dict[str, list[Exception]]):
        self.pool = errors

    def _format_error_message(self):
        """Format the error message for the UIJsonError."""

        msg = ""
        for key, errors in self.pool.items():
            if errors:
                msg += f"\t{key}:\n"
                for i, error in enumerate(errors):
                    msg += f"\t\t{i}. {error}\n"

        return msg

    def throw(self):
        """Raise the UIJsonError with detailed list of errors per parameter."""

        message = self._format_error_message()
        if message:
            message = "Collected UIJson errors:\n" + message
            raise UIJsonError(message)


def dependency_type_validation(name: str, ui_json: BaseUIJson):
    """
    Validate that the form depending on is optional, group_optional or bool type.

    :param name: Name of the form
    :param ui_json: A UIJson object.
    """
    dependency_form = getattr(ui_json, name)

    if not (
        dependency_form.optional
        or isinstance(dependency_form, BoolForm)
        or dependency_form.group_optional
    ):
        raise UIJsonError(
            f"Dependency form '{name}' must be either optional, group_optional or of boolean type."
        )


def get_validations(form: list[str]) -> list[Callable]:
    """
    Get callable validations based on identifying form keys.

    :param form: Form to validate sub-keys with

    :return: List of callable validations.
    """
    return [VALIDATIONS_MAP[k] for k in form if k in VALIDATIONS_MAP]


def mesh_type_validation(name: str, data: dict[str, Any], ui_json: BaseUIJson):
    """
    Validate that value is one of the provided mesh types.

    :param name: Name of the form
    :param data: Input data with known validations.
    :param ui_json: A UIJson object.
    """

    form = getattr(ui_json, name)
    mesh_types = form.mesh_type

    obj = data[name]
    if not isinstance(obj, tuple(mesh_types)):
        raise UIJsonError(f"Object's mesh type must be one of {mesh_types}.")


def parent_validation(name: str, data: dict[str, Any], ui_json: BaseUIJson):
    """
    Validate that the data is a child of the parent object.

    :param name: Name of the form
    :param data: Input data with known validations.
    :param ui_json: A UIJson object.
    """

    form = getattr(ui_json, name)
    if isinstance(form, DataOrValueForm) and form.is_value:
        return

    parent_name = form.parent
    child = data[name]

    # Special case for DataRangeForm
    if isinstance(child, dict):
        child = data[name]["property"]

    parent = data[parent_name]

    child = child if isinstance(child, list) else [child]
    if (
        not isinstance(parent, ObjectBase)
        or len(list(set(child) - set(parent.children))) > 0
    ):
        raise UIJsonError(f"{name} data is not a child of {parent_name}.")


def promote_or_catch(
    workspace: Workspace,
    value: Any,
) -> Any:
    """
    Returns an object if it exists in the workspace or an error if not.

    :param workspace: Workspace to fetch entities from.
    :param value: UUID of the object to fetch.

    :returns: The object if it exists in the workspace or a placeholder error
        to be collected and raised later with any other UIJson level validation
        errors.
    """
    if isinstance(value, list | tuple):
        return [promote_or_catch(workspace, val) for val in value]

    if isinstance(value, dict):
        return {key: promote_or_catch(workspace, val) for key, val in value.items()}

    if not isinstance(value, UUID):
        return value

    obj = workspace.get_entity(value)
    if obj[0] is not None:
        return obj[0]

    raise UIJsonError(f"Workspace does not contain an entity with uid: {value}.")


VALIDATIONS_MAP = {
    "mesh_type": mesh_type_validation,
    "parent": parent_validation,
}
