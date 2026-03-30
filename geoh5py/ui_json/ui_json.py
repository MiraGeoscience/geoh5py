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
    create_model,
    field_validator,
)

from geoh5py import Workspace
from geoh5py.groups import UIJsonGroup
from geoh5py.shared.utils import (
    as_str_if_uuid,
    dict_mapper,
    entity2uuid,
    fetch_active_workspace,
)
from geoh5py.ui_json.annotations import OptionalPath
from geoh5py.ui_json.forms import BaseForm
from geoh5py.ui_json.validation import (
    ErrorPool,
    UIJsonError,
    get_validations,
    promote_or_catch,
)


logger = logging.getLogger(__name__)


class UIJson(BaseModel):
    """
    Base class for storing ui.json data on disk.

    :param version: Version of the application.
    :params title: Title of the application.
    :params geoh5: Path to the geoh5 file.
    :params run_command: Command to run the application.
    :params monitoring_directory: Directory to monitor for changes.
    :params conda_environment: Conda environment to run the application.
    :params workspace_geoh5: Path to the workspace geoh5 file.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, extra="allow", validate_assignment=True
    )

    version: str
    title: str
    geoh5: Path | None
    run_command: str
    monitoring_directory: OptionalPath
    conda_environment: str
    workspace_geoh5: OptionalPath | None = None
    _path: Path | None = None
    _groups: dict[str, list[str]]

    def model_post_init(self, context: Any, /) -> None:
        self._groups = self.get_groups()

    def __repr__(self) -> str:
        """Repr level shows a path if it exists or the title otherwise."""
        return f"UIJson('{self.title if self._path is None else str(self._path.name)}')"

    def __str__(self) -> str:
        """String level shows the full json representation."""

        json_string = self.model_dump_json(indent=4, exclude_unset=True)

        for field in self.model_fields.keys():
            value = getattr(self, field)
            if isinstance(value, BaseForm):
                type_string = type(value).__name__
                json_string = json_string.replace(
                    f'"{field}": {{', f'"{field}": {type_string} {{'
                )

        return f"{self!r} -> {json_string}"

    @field_validator("geoh5", mode="after")
    @classmethod
    def workspace_path_exists(cls, path: Path | None) -> Path | None:
        if path is not None and not path.exists():
            raise FileNotFoundError(f"geoh5 path {path} does not exist.")
        return path

    @field_validator("geoh5", mode="after")
    @classmethod
    def valid_geoh5_extension(cls, path: Path | None) -> Path | None:
        if path is not None and path.suffix != ".geoh5":
            raise ValueError(
                f"Workspace path: {path} must have a '.geoh5' file extension."
            )
        return path

    @staticmethod
    def load(path: str | Path) -> dict:
        """
        Load ui json from file.

        :param path: Path to the .ui.json file.

        :return: Dictionary representing the ui json object.
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
            kwargs = {
                key: (item if item != "" else None) for key, item in kwargs.items()
            }

        return kwargs

    @classmethod
    def read(cls, path: str | Path) -> UIJson:
        """
        Create a UIJson object from ui.json file.

        Raises errors if the file doesn't exist or is not a .ui.json file.
        Also validates at the Form and UIJson level whether the file is
        properly formatted.  If called from the BaseUIJson class, forms
        will be inferred dynamically.

        :param path: Path to the .ui.json file.
        :returns: UIJson object.
        """

        kwargs = cls.load(path)

        fields = {}
        for name, value in kwargs.items():
            if name in UIJson.model_fields.keys():
                continue
            if isinstance(value, dict):
                form_type = BaseForm.infer(value)
                fields[name] = (form_type, ...)
            else:
                fields[name] = (type(value), ...)

        model = create_model(  # type: ignore
            kwargs.get("title", "UnknownUIJson"),
            __base__=UIJson,
            **fields,
        )
        uijson = model(**kwargs)

        uijson._path = path  # pylint: disable=protected-access
        return uijson

    def write(self, path: Path):
        """
        Write the UIJson object to file.

        :param path: Path to write the .ui.json file.
        """
        self._path = path
        with open(path, "w", encoding="utf-8") as file:
            data = self.model_dump_json(indent=4, exclude_unset=True, by_alias=True)
            file.write(data)

    def get_groups(self) -> dict[str, list[str]]:
        """
        Returns grouped forms.

        :returns: Group names and the parameters belonging to each
            group.
        """
        groups: dict[str, list[str]] = {}
        for field in self.__class__.model_fields.keys():
            form = getattr(self, field)
            if not isinstance(form, BaseForm):
                continue
            name = getattr(form, "group", "")
            if name:
                groups[name] = [field] if name not in groups else groups[name] + [field]

        return groups

    def is_disabled(self, field: str) -> bool:
        """
        Checks if a field is disabled based on form status.

        :param field: Field name to check.
        :returns: True if the field is disabled by its own enabled status or
            the groups enabled status, False otherwise.
        """

        value = getattr(self, field)
        if not isinstance(value, BaseForm):
            return False
        if value.enabled is False:
            return True

        disabled = False
        if value.group:
            group = next(v for k, v in self._groups.items() if field in v)
            for member in group:
                form = getattr(self, member)
                if form.group_optional:
                    disabled = not form.enabled
                    break

        return disabled

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
            if skip_disabled and self.is_disabled(field):
                continue

            value = getattr(self, field)
            if isinstance(value, BaseForm):
                value = value.flatten()
            data[field] = value

        return data

    def set_values(self, copy: bool = False, **kwargs) -> UIJson:
        """
        Fill the UIJson with new values.

        :param copy: If True, returns a new UIJson object with the updated values.
            If False, updates the current UIJson object with the new values and returns itself.
        :param kwargs: Key/value pairs to update the UIJson with.

        :return: A new UIJson object with the updated values.
        """
        if copy:
            uijson = self.model_copy(deep=True)
        else:
            uijson = self

        demotion = [entity2uuid, as_str_if_uuid]
        for key, value in kwargs.items():
            form = getattr(uijson, key, None)
            if isinstance(form, BaseForm):
                form.set_value(value)
            else:
                setattr(uijson, key, dict_mapper(value, demotion))

        return uijson

    def to_params(self, workspace: Workspace | None = None) -> dict[str, Any]:
        """
        Promote, flatten and validate parameter/values dictionary.

        :param workspace: Workspace to fetch entities from.  Used for passing active
            workspaces to avoid closing and flushing data.

        :returns: If the data passes validation, to_params returns a promoted and
            flattened parameters/values dictionary that may be dumped into an application
            specific params (options) class.
        """

        data = self.flatten(skip_disabled=True, active_only=True)
        with fetch_active_workspace(workspace or Workspace(self.geoh5)) as geoh5:
            if geoh5 is None:
                raise ValueError("Workspace cannot be None.")

            errors: dict[str, Any] = {k: [] for k in self.model_fields_set}
            for field, value in data.items():
                if field == "geoh5":
                    data[field] = geoh5
                    continue

                try:
                    value = promote_or_catch(geoh5, value)
                except UIJsonError:
                    errors[field].append(value)

                data[field] = value

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

        ui_json = self.model_dump(exclude_unset=True)
        for field, form in ui_json.items():
            if self.is_disabled(field):
                continue

            validations = get_validations(list(form) if isinstance(form, dict) else [])
            for validation in validations:
                try:
                    validation(field, params, ui_json)
                except UIJsonError as e:
                    errors[field].append(e)

        ErrorPool(errors).throw()
