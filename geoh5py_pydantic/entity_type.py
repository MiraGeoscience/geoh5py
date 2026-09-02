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

from __future__ import annotations

from typing import Any, ClassVar
from uuid import UUID, uuid4

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class NamedIdentity(BaseModel):
    """Common aliased identity fields for entities and entity types."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )

    uid: UUID = Field(
        default_factory=uuid4,
        validation_alias=AliasChoices("uid", "ID"),
        serialization_alias="ID",
    )
    name: str = Field(
        default="Entity",
        validation_alias=AliasChoices("name", "Name"),
        serialization_alias="Name",
    )


class EntityType(NamedIdentity):
    """
    Workspace-free identity and attributes for a shared geoh5 entity type.

    Concrete type categories declare their HDF5 collections once, while each
    instance carries the UID, name, and description written to its type group.
    """

    h5_collection: ClassVar[str | None] = None
    h5_type_collection: ClassVar[str | None] = None

    description: str | None = Field(
        default="Entity",
        validation_alias=AliasChoices("description", "Description"),
        serialization_alias="Description",
    )

    def h5_attributes(self) -> dict[str, Any]:
        """Return the values stored as attributes on the shared type group."""
        return self.model_dump(by_alias=True, exclude_none=True)


class DataType(EntityType):
    """HDF5 placement shared by future data type models."""

    h5_collection: ClassVar[str] = "Data"
    h5_type_collection: ClassVar[str] = "Data types"


class GroupType(EntityType):
    """HDF5 placement and common attributes for group type models."""

    h5_collection: ClassVar[str] = "Groups"
    h5_type_collection: ClassVar[str] = "Group types"

    # These attributes are specific to legacy GroupType and are stored on
    # every shared group type, including the type used by the project Root.
    allow_move_content: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "allow_move_content",
            "Allow move contents",
        ),
        serialization_alias="Allow move contents",
    )
    allow_delete_content: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "allow_delete_content",
            "Allow delete contents",
        ),
        serialization_alias="Allow delete contents",
    )


class ObjectType(EntityType):
    """HDF5 placement shared by object types such as Points."""

    h5_collection: ClassVar[str] = "Objects"
    h5_type_collection: ClassVar[str] = "Object types"
