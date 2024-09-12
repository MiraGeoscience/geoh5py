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

import json
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator

from geoh5py import Workspace
from geoh5py.shared.utils import fetch_active_workspace
from geoh5py.ui_json.forms import BaseForm
from geoh5py.ui_json.validation import ErrorLane, ErrorPool, UIJsonError


class BaseUIJson(BaseModel):
    """
    Base class for storing ui.json data on disk.

    :params title: Title of the application.
    :params geoh5: Path to the geoh5 file.
    :params run_command: Command to run the application.
    :params run_command_boolean: Boolean to run the command.
    :params monitoring_directory: Directory to monitor for changes.
    :params conda_environment: Conda environment to run the application.
    :params conda_environment_boolean: Boolean to run the conda environment.
    :params workspace_geoh5: Path to the workspace geoh5 file.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    title: str
    geoh5: Path | None
    run_command: str
    monitoring_directory: Path
    conda_environment: str
    workspace_geoh5: Path

    @field_validator("workspace_geoh5", mode="after")
    @classmethod
    def current_directory_if_workspace_doesnt_exist(cls, path):
        if not path.exists():
            return Path()
        return path

    @field_validator("geoh5", mode="after")
    @classmethod
    def workspace_path_exists(cls, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"geoh5 path {path} does not exist.")
        return path

    @classmethod
    def read(cls, path: Path):
        """Create a UIJson object from file."""

        path = path.resolve()

        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist.")

        if "".join(path.suffixes[-2:]) != ".ui.json":
            raise ValueError(f"File {path} is not a .ui.json file.")

        with open(path, encoding="utf-8") as file:
            kwargs = json.load(file)

        return cls(**kwargs)

    def to_params(self, workspace: Workspace | None = None) -> dict[str, Any]:
        """
        Promote, flatten and validate parameter/values dictionary.

        :param workspace: Workspace to fetch entities from.  Used for passing active
            workspaces to avoid closing and flushing data.
        """

        with fetch_active_workspace(workspace or Workspace(self.geoh5)) as geoh5:
            if geoh5 is None:
                raise ValueError("Workspace cannot be None.")

            data = {}
            for field in self.model_fields_set:
                if field == "geoh5":
                    data[field] = geoh5
                    continue

                value = getattr(self, field)
                value = value.flatten() if isinstance(value, BaseForm) else value

                if isinstance(value, UUID):
                    value = self._object_or_catch(geoh5, value)

                data[field] = value

            self.validate_data(data)

        return data

    def validate_data(self, params: dict[str, Any] | None = None):
        """
        Validate the UIJson data.

        :param params: promoted and flattened parameters/values dictionary.  The params
            dictionary will be generated from the model values if not provided.

        :raises UIJsonError: If any validations fail.
        """

        if params is None:
            self.to_params()
            return

        for field in self.model_fields_set:
            value = getattr(self, field)
            if not isinstance(value, BaseForm):
                continue

            try:
                value.validate_data(params)
            except UIJsonError as e:
                if isinstance(params[field], ErrorLane):
                    params[field].catch(e)
                else:
                    params[field] = ErrorLane(e)

        error_pool = ErrorPool(
            {k: v for k, v in params.items() if isinstance(v, ErrorLane)}
        )
        error_pool.throw()

    def _object_or_catch(
        self,
        workspace: Workspace,
        uuid: UUID,
    ):
        """
        Returns an object if it exists in the workspace or an error if not.

        :param workspace: Workspace to fetch entities from.
        :param uuid: UUID of the object to fetch.
        """

        obj = workspace.get_entity(uuid)
        if obj[0] is not None:
            return obj[0]

        error_lane = ErrorLane(
            UIJsonError(f"Workspace does not contain an entity with uid: {uuid}.")
        )

        return error_lane
