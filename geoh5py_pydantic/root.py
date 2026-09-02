# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2026 Mira Geoscience Ltd.                                     '
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

"""Pydantic models for the mandatory root entity in every geoh5 file."""

from __future__ import annotations

from typing import ClassVar
from uuid import UUID

from pydantic import AliasChoices, Field, field_validator

from .entity import Attributes, PydanticEntity
from .entity_type import GroupType


# the fixed UID used by the legacy NoTypeGroup
ROOT_TYPE_UID = UUID("{dd99b610-be92-48c0-873c-5b5946ea2840}")


class RootAttributes(Attributes):
    """
    Core entity attributes with the defaults used by legacy RootGroup.
    Its name is ``"Workspace"`` and moving, deleting, and renaming are disabled.
    """

    name: str = Field(
        default="Workspace",
        validation_alias=AliasChoices("name", "Name"),
        serialization_alias="Name",
    )
    allow_delete: bool = Field(
        default=False,
        validation_alias=AliasChoices("allow_delete", "Allow delete"),
        serialization_alias="Allow delete",
    )
    allow_move: bool = Field(
        default=False,
        validation_alias=AliasChoices("allow_move", "Allow move"),
        serialization_alias="Allow move",
    )
    allow_rename: bool = Field(
        default=False,
        validation_alias=AliasChoices("allow_rename", "Allow rename"),
        serialization_alias="Allow rename",
    )


class RootType(GroupType):
    """The fixed no-type GroupType linked from the project Root."""

    uid: UUID = Field(
        default=ROOT_TYPE_UID,
        validation_alias=AliasChoices("uid", "ID"),
        serialization_alias="ID",
    )
    name: str = Field(
        default="Workspace",
        validation_alias=AliasChoices("name", "Name"),
        serialization_alias="Name",
    )
    description: str | None = Field(
        default="Entity",
        validation_alias=AliasChoices("description", "Description"),
        serialization_alias="Description",
    )

    @field_validator("uid")
    @classmethod
    def validate_uid(cls, value: UUID) -> UUID:
        """The Root type UID is part of the geoh5 format contract."""
        if value != ROOT_TYPE_UID:
            raise ValueError(f"Root type UID must be {ROOT_TYPE_UID}.")
        return value


class Root(PydanticEntity):
    """Workspace-free representation of the mandatory geoh5 Root group."""

    attributes_model: ClassVar[type[Attributes]] = RootAttributes

    attributes: RootAttributes = Field(
        default_factory=RootAttributes,
        validation_alias=AliasChoices("attributes", "attrs"),
        serialization_alias="attrs",
    )
    entity_type: RootType = Field(default_factory=RootType)
    parent_uid: None = None
