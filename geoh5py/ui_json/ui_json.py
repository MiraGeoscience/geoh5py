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

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    PrivateAttr,
    create_model,
    field_validator,
)

from geoh5py import Workspace
from geoh5py.groups import UIJsonGroup
from geoh5py.shared.utils import (
    copy_dict_relatives,
    dict_mapper,
    entity2uuid,
    fetch_active_workspace,
)
from geoh5py.ui_json.annotations import OptionalPath, OptionalString
from geoh5py.ui_json.forms import BaseForm, DependencyType, GroupForm
from geoh5py.ui_json.validation import (
    ErrorPool,
    UIJsonError,
    dependency_type_validation,
    get_validations,
    promote_or_catch,
)


logger = logging.getLogger(__name__)


class BaseUIJson(BaseModel):
    """
    Base class for storing ui.json data on disk.

    :param version: Version of the application.
    :params title: Title of the application.
    :params geoh5: Path to the geoh5 file.
    :params run_command: Command to run the application.
    :params monitoring_directory: Directory to monitor for changes.
    :params conda_environment: Conda environment to run the application.
    :params workspace_geoh5: Path to the workspace geoh5 file.
    :params out_group: Optional group form to hold the UIJson group.

    :params _form_dependencies: Nested dictionaries describing the dependencies between forms,
        where the key is the name of the form, and the value is a dictionary of
        forms name and respective mirroring enabled state behaviour
        (True: reflects, False: reverses).

    :param _group_dependencies: Dictionary holding the name of the groups and
        leading form.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, extra="allow", validate_assignment=True
    )

    version: str | None = "0.0.0"
    title: str
    geoh5: OptionalPath
    run_command: str | None
    monitoring_directory: OptionalPath = None
    conda_environment: str | None
    workspace_geoh5: OptionalPath = None

    out_group: GroupForm | OptionalString = None

    _form_dependencies: dict[str, dict[str, bool]] = PrivateAttr(default_factory=dict)
    _group_dependencies: dict[str, BaseForm] = PrivateAttr(default_factory=dict)

    def copy_relatives(self, parent: Workspace, clear_cache: bool = False):
        """
        Copy the entities referenced in the input file to a new workspace.

        :param parent: The parent to copy the entities to.
        :param clear_cache: Indicate whether to clear the cache.
        """
        if self.geoh5 is None:
            logger.warning("No geoh5 file path set; nothing to copy.")
            return

        with Workspace(self.geoh5, mode="r") as geoh5:
            params = self.to_params(workspace=geoh5)
            params.pop("geoh5", None)
            copy_dict_relatives(
                params,
                parent,
                clear_cache=clear_cache,
            )

    def flatten(self, skip_disabled=False, active_only=False) -> dict[str, Any]:
        """
        Flatten the UIJson data to dictionary of key/value pairs.

        Chooses between value/property in data forms depending on the is_value
        field.

        :param skip_disabled: If True, skips fields with 'enabled' set to False.
        :param active_only: If True, skips fields that have not been explicitly set.

        :return: Flattened dictionary of key/value pairs.
        """
        data = {}
        fields = self.model_fields_set if active_only else self.model_fields

        for field in fields:
            if skip_disabled and not self.is_enabled(field):
                continue

            value = getattr(self, field)
            if isinstance(value, BaseForm):
                value = value.flatten()
            data[field] = value

        return data

    @classmethod
    def from_dict(cls, data: dict) -> BaseUIJson:
        """
        Create a UIJson instance from a dictionary.

        :param data: Dictionary representing the ui json object.

        :returns: UIJson object.
        """
        kwargs = {key: (item if item != "" else None) for key, item in data.items()}

        ui_json_class = cls.infer(**kwargs)

        return ui_json_class(**kwargs)

    @property
    def form_dependencies(self) -> dict[str, dict[str, bool]]:
        """Stashed inter-form dependencies."""
        return self._form_dependencies

    @staticmethod
    def infer(title="UnknownUIJson", **kwargs) -> type[BaseUIJson]:
        """
        Create a UIJson subclass dynamically based on inferred form types.

        For each keyword argument that is not already a field on :class:`UIJson`,
        the function tries to infer the appropriate :class:`~geoh5py.ui_json.forms.BaseForm`
        subclass from the value if it is a dict, or uses the value's type directly
        otherwise.

        :param title: Name for the generated model class.  Defaults to ``"UnknownUIJson"``.
        :param kwargs: Named form data to include in the generated class.

        :return: A new :class:`UIJson` subclass whose extra fields match the inferred types.
        """
        fields = {}
        for name, value in kwargs.items():
            if name in BaseUIJson.model_fields.keys():
                continue
            if isinstance(value, dict):
                form_type = BaseForm.infer(value)
                fields[name] = (form_type, ...)
            else:
                fields[name] = (type(value), ...)

        model = create_model(  # type: ignore
            kwargs.get("title", title),
            __base__=BaseUIJson,
            **fields,
        )
        return model

    def is_enabled(self, field: str) -> bool:
        """
        Checks if a field is enabled based on form status and linkages.

        :param field: Field name or form to check.
        :returns: False if the field is disabled by its own enabled status or
            the linkages enabled status, True otherwise.
        """
        enabled = True
        form = getattr(self, field)

        # Only a key:value pair, cannot be disabled
        if not isinstance(form, BaseForm):
            return True

        if not form.enabled:
            return False

        # Can still be disabled based on linkages
        # Check if disabled based on group status
        group = getattr(form, "group", "")
        if group in self._group_dependencies:
            enabled = self._group_dependencies[group].enabled

        # Then check on direct dependency
        for name, mirror in self._form_dependencies[field].items():
            # Not enabled as soon as False is encountered
            if not enabled:
                return False

            codependent = getattr(self, name)

            if mirror:
                enabled = codependent.enabled
            else:
                enabled = not codependent.enabled

        return enabled

    def model_post_init(self, context: Any, /) -> None:
        self._group_dependencies, self._form_dependencies = self._get_dependency_links()

    @classmethod
    def read(cls, path: str | Path) -> BaseUIJson:
        """
        Create a UIJson instance from ui.json file.

        Raises errors if the file doesn't exist or is not a .ui.json file.
        Also validates at the Form and UIJson level whether the file is
        properly formatted.

        Consider using the `load` method to get the UIJson class and data separately
        if you want to handle validation errors yourself.

        :param path: Path to the .ui.json file.

        :returns: UIJson object.
        """
        kwargs = cls._load(path)

        return cls.from_dict(kwargs)

    def set_enabled(self, copy: bool = False, **states) -> BaseUIJson:
        """
        Set the enabled state of fields, and handle the state of dependencies.

        :param copy: If True, returns a new UIJson object with the updated values.
            If False, updates the current UIJson object with the new values and returns itself.
        :param states: Dictionary of field names and their enabled state to update.

        :return: A UIJson object with the updated values.
        """
        if copy:
            uijson = self.model_copy(deep=True)
        else:
            uijson = self

        for field, value in states.items():
            form = getattr(uijson, field, None)
            if not isinstance(form, BaseForm):
                continue

            if not value and not form.is_optional:
                raise ValueError(f"Field {field} enabled state cannot be False.")

            form.enabled = value

            # Mirror the state to dependencies
            for name, mirror in uijson.form_dependencies[field].items():
                # Set the link dependency state
                codependent = getattr(uijson, name)

                if mirror:
                    codependent.enabled = value
                else:
                    codependent.enabled = not value

        return uijson

    def set_values(self, copy: bool = False, **kwargs) -> BaseUIJson:
        """
        Fill the UIJson with new values.

        :param copy: If True, returns a new UIJson object with the updated values.
            If False, updates the current UIJson object with the new values and returns itself.
        :param kwargs: Key/value pairs to update the UIJson with.

        :return: A UIJson object with the updated values.
        """
        if copy:
            uijson = self.model_copy(deep=True)
        else:
            uijson = self

        for field, value in kwargs.items():
            form = getattr(uijson, field, None)
            if isinstance(form, BaseForm):
                if not (form.is_optional and value is None):
                    form.set_value(value)

                uijson.set_enabled(copy=False, **{field: value is not None})
            else:
                setattr(uijson, field, dict_mapper(value, [entity2uuid]))

        return uijson

    def to_params(
        self, workspace: Workspace | None = None, validate=True
    ) -> dict[str, Any]:
        """
        Promote, flatten and validate parameter/values dictionary.

        :param workspace: Workspace to fetch entities from.  Used for passing active
            workspaces to avoid closing and flushing data.
        :param validate: Whether to run cross validations on the data after

        :returns: A flattened parameters/values dictionary that may be dumped into an application
            specific params (options) class. If validate=True, the content is validated and errors
            are raised if any validations fail.
        """
        data = self.flatten(skip_disabled=True, active_only=True)

        with (
            fetch_active_workspace(workspace)
            if workspace
            else Workspace(self.geoh5, mode="r")
        ) as geoh5:
            if geoh5 is None:
                raise ValueError("Workspace cannot be None.")

            errors: dict[str, Any] = {k: [] for k in self.model_fields_set}
            for field, value in data.items():
                if field == "geoh5":
                    data[field] = geoh5
                    continue

                try:
                    value = promote_or_catch(geoh5, value)
                except UIJsonError as e:
                    errors[field].append(e)

                data[field] = value

            if validate:
                self._cross_validations(data, errors)

        return data

    def to_ui_json_group(
        self, workspace: Workspace | None = None, **kwargs
    ) -> UIJsonGroup:
        """
        Convert the UIJson to a UIJsonGroup.

        :param workspace: Workspace to fetch entities from.  Used for passing active
            workspaces to avoid closing and flushing data.
        :param kwargs: Additional keyword arguments to update the UIJson data before

        :return: A UIJsonGroup representing the application.
        """
        with fetch_active_workspace(workspace or Workspace(self.geoh5)) as geoh5:
            if geoh5 is None:
                raise ValueError("Workspace cannot be None.")

            ui_json_group = UIJsonGroup.create(
                workspace=geoh5,
                options=self.model_dump(mode="json", exclude_unset=True, by_alias=True),
                name=kwargs.pop("name", self.title),
                **kwargs,
            )
            options = ui_json_group.options
            options["out_group"]["value"] = ui_json_group.uid
            options["out_group"]["enabled"] = True
            ui_json_group.options = options

            return ui_json_group

    @field_validator("geoh5", mode="after")
    @classmethod
    def valid_geoh5_extension(cls, path: Path | None) -> Path | None:
        """
        Check if the input has a valid geoh5 extension.

        :param path: Path to the file to check.
        :return: Return Path if provided
        """
        if path is not None and path.suffix != ".geoh5":
            raise ValueError(
                f"Workspace path: {path} must have a '.geoh5' file extension."
            )
        return path

    @field_validator("geoh5", mode="after")
    @classmethod
    def workspace_path_exists(cls, path: Path | None) -> Path | None:
        """
        Check if the workspace path exists.

        :param path: Path to the file to check.
        :return: Return Path if provided
        """
        if path is not None and not path.exists():
            raise FileNotFoundError(f"geoh5 path {path} does not exist.")
        return path

    def write(self, path: Path) -> Path:
        """
        Write the UIJson object to file.

        :param path: Path to write the .ui.json file.

        :return: Return path to the ui_json file.
        """
        with open(path, "w", encoding="utf-8") as file:
            data = self.model_dump_json(indent=4, exclude_unset=True, by_alias=True)
            file.write(data)

        return path

    def _cross_validations(
        self, params: dict[str, Any], errors: dict[str, Any] | None = None
    ) -> None:
        """
        Extra validation related to inter-form dependencies and entity types.

        :param params: Promoted and flattened parameters/values dictionary.  The params
            dictionary will be generated from the model values if not provided.
        :param errors: Optionally pass existing errors. Primarily for the to_params
            method.

        :raises UIJsonError: If any validations fail.
        """
        if errors is None:
            errors = {k: [] for k in params}

        for field in self.model_fields_set:
            if not self.is_enabled(field):
                continue
            form = getattr(self, field)
            validations = get_validations(
                list(form.model_fields_set) if isinstance(form, BaseForm) else []
            )
            for validation in validations:
                try:
                    validation(field, params, self)
                except UIJsonError as e:
                    errors[field].append(e)

        ErrorPool(errors).throw()

    def _get_dependency_links(
        self,
    ) -> tuple[dict[str, BaseForm], dict[str, dict[str, bool]]]:
        """
        Returns dependency links between forms.

        For each form, there can be a group dependency ('group') to a leading
        form ('group_optional') and/or a direct dependency between forms ('dependency').

        A direct dependency controls the enabled state two ways, while the group dependency
        controls the enabled state only from the lead form to its dependents.

        :returns: Tuple of group dependencies and direct form dependencies.
        """
        form_dependencies: dict[str, dict[str, bool]] = {}
        group_dependencies: dict[str, BaseForm] = {}

        for name in self.__class__.model_fields.keys():
            if name not in form_dependencies:
                form_dependencies[name] = {}

            form = getattr(self, name)

            if not isinstance(form, BaseForm):
                continue

            # Check for groupOptional dependency
            # Only the leading form should have the groupOptional field
            group_name: str = getattr(form, "group", "")
            group_optional = getattr(form, "group_optional", False)
            if group_optional:
                group_dependencies[group_name] = form

            # Check for direct dependency on other form
            dependents_on = form.dependency

            # If optional, enabled state only influences the form
            if dependents_on and not getattr(form, "optional", False):
                dependency_type_validation(dependents_on, self)
                mirrors = getattr(form, "dependency_type", None) in [
                    DependencyType.ENABLED,
                    DependencyType.SHOW,
                ]

                # Add reverse linkage
                if dependents_on not in form_dependencies:
                    form_dependencies[dependents_on] = {}

                form_dependencies[dependents_on].update({name: mirrors})
                form_dependencies[name][dependents_on] = mirrors

        return group_dependencies, form_dependencies

    @staticmethod
    def _load(path: str | Path) -> dict:
        """
        Load data and generate a UIJson class from file.

        :param path: Path to the .ui.json file.

        :return: UIJson class and dictionary representing the ui json object.
        """
        if isinstance(path, str):
            path = Path(path)

        path = path.resolve()

        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist.")

        if "".join(path.suffixes[-2:]) != ".ui.json":
            raise ValueError(f"File {path} is not a .ui.json file.")

        with open(path, encoding="utf-8") as file:
            kwargs = json.load(file)

        return kwargs

    def __repr__(self) -> str:
        """Repr level shows the title."""
        return f"UIJson('{self.title}')"

    def __str__(self) -> str:
        """String level shows the full json representation."""

        json_string = self.model_dump_json(indent=4, exclude_unset=True)
        for field in type(self).model_fields.keys():
            value = getattr(self, field)
            if isinstance(value, BaseForm):
                type_string = type(value).__name__
                json_string = json_string.replace(
                    f'"{field}": {{', f'"{field}": {type_string} {{'
                )

        return f"{self!r} -> {json_string}"
