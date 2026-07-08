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

from typing import Any, Self
from uuid import UUID, uuid4

import numpy as np
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


class PydanticEntity(BaseModel):
    """
    Workspace-free base model for geoh5 entity attributes.

    This class deliberately models identity and metadata only. Persistence,
    parent/child registration, and workspace mutation might live in adapters
    around this model rather than in the model itself.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
        populate_by_name=True,
        validate_assignment=True,
    )

    uid: UUID = Field(default_factory=uuid4, validation_alias=AliasChoices("uid", "ID"))
    name: str = Field(default="Entity", validation_alias=AliasChoices("name", "Name"))
    type_uid: UUID | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "type_uid", "Type ID", "Object Type ID", "Data Type ID"
        ),
    )
    parent_uid: UUID | None = None
    allow_delete: bool = Field(
        default=True, validation_alias=AliasChoices("allow_delete", "Allow delete")
    )
    allow_move: bool = Field(
        default=True, validation_alias=AliasChoices("allow_move", "Allow move")
    )
    allow_rename: bool = Field(
        default=True, validation_alias=AliasChoices("allow_rename", "Allow rename")
    )
    clipping_ids: list[UUID] | None = Field(
        default=None, validation_alias=AliasChoices("clipping_ids", "Clipping IDs")
    )
    metadata: dict[str, Any] | None = Field(
        default=None, validation_alias=AliasChoices("metadata", "Metadata")
    )
    on_file: bool = False
    partially_hidden: bool = Field(
        default=False,
        validation_alias=AliasChoices("partially_hidden", "Partially hidden"),
    )
    public: bool = Field(
        default=True, validation_alias=AliasChoices("public", "Public")
    )
    visible: bool = Field(
        default=True, validation_alias=AliasChoices("visible", "Visible")
    )

    @field_validator("metadata", mode="before")
    @classmethod
    def validate_metadata(cls, value: Any) -> dict[str, Any] | None:
        """
        Keep the common geoh5 metadata coercions local to the entity model.
        """
        if isinstance(value, np.ndarray):
            value = value[0]

        if isinstance(value, bytes):
            value = value.decode("utf-8")

        if isinstance(value, str):
            # Leave JSON parsing to a later adapter; this keeps the skeleton
            # independent from the existing shared utility module.
            return {"raw": value}

        if value is not None and not isinstance(value, dict):
            raise ValueError(
                "Input metadata must be of type dict, bytes, str, ndarray or None."
            )

        return value

    @classmethod
    def from_legacy_entity(cls, entity: Any, **overrides) -> Self:
        """
        Build a pydantic model from an existing geoh5py Entity-like object.
        """
        parent = getattr(entity, "parent", None)
        attrs = {
            "uid": getattr(entity, "uid", None),
            "name": getattr(entity, "name", None),
            "type_uid": getattr(getattr(entity, "entity_type", None), "uid", None),
            "parent_uid": getattr(parent, "uid", None),
            "allow_delete": getattr(entity, "allow_delete", True),
            "allow_move": getattr(entity, "allow_move", True),
            "allow_rename": getattr(entity, "allow_rename", True),
            "clipping_ids": getattr(entity, "clipping_ids", None),
            "metadata": getattr(entity, "metadata", None),
            "on_file": getattr(entity, "on_file", False),
            "partially_hidden": getattr(entity, "partially_hidden", False),
            "public": getattr(entity, "public", True),
            "visible": getattr(entity, "visible", True),
        }
        attrs = {key: value for key, value in attrs.items() if value is not None}
        attrs.update(overrides)

        return cls.model_validate(attrs)

    def model_dump_geoh5_attributes(self) -> dict[str, Any]:
        """
        Dump the core entity attributes using geoh5 naming.
        """
        return {
            "ID": self.uid,
            "Name": self.name,
            "Allow delete": self.allow_delete,
            "Allow move": self.allow_move,
            "Allow rename": self.allow_rename,
            "Clipping IDs": self.clipping_ids,
            "Metadata": self.metadata,
            "Partially hidden": self.partially_hidden,
            "Public": self.public,
            "Visible": self.visible,
        }
