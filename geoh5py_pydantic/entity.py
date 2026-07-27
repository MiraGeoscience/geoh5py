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

from typing import Annotated, Any, ClassVar, Self
from uuid import UUID, uuid4

import numpy as np
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from .entity_type import EntityType, NamedIdentity


class Attributes(NamedIdentity):
    """
    Core HDF5 attributes for objects and groups.

    Python field names remain convenient in model code. Serialization aliases
    are the exact attribute names stored in a geoh5 file.
    """

    allow_delete: bool = Field(
        default=True,
        validation_alias=AliasChoices("allow_delete", "Allow delete"),
        serialization_alias="Allow delete",
    )
    allow_move: bool = Field(
        default=True,
        validation_alias=AliasChoices("allow_move", "Allow move"),
        serialization_alias="Allow move",
    )
    allow_rename: bool = Field(
        default=True,
        validation_alias=AliasChoices("allow_rename", "Allow rename"),
        serialization_alias="Allow rename",
    )
    clipping_ids: list[UUID] | None = Field(
        default=None,
        validation_alias=AliasChoices("clipping_ids", "Clipping IDs"),
        serialization_alias="Clipping IDs",
    )
    last_focus: str | None = Field(
        validation_alias=AliasChoices("last_focus", "Last focus"),
        serialization_alias="Last focus",
    )
    name: str = Field(
        default="Entity",
        validation_alias=AliasChoices("name", "Name"),
        serialization_alias="Name",
    )
    partially_hidden: bool = Field(
        default=False,
        validation_alias=AliasChoices("partially_hidden", "Partially hidden"),
        serialization_alias="Partially hidden",
    )
    public: bool = Field(
        default=True,
        validation_alias=AliasChoices("public", "Public"),
        serialization_alias="Public",
    )
    uid: UUID = Field(
        default=uuid4(),
        validation_alias=AliasChoices("uid", "ID"),
        serialization_alias="ID",
    )
    visible: bool = Field(
        default=True,
        validation_alias=AliasChoices("visible", "Visible"),
        serialization_alias="Visible",
    )


class PydanticEntity(BaseModel):
    """
    Workspace-free entity with nested attributes and shared type information.

    Persistence, parent/child registration, and file mutation remain outside
    the model. Convenience properties retain direct access such as
    ``entity.name`` while ``entity.attributes`` owns the serialized fields.
    """

    attributes_model: ClassVar[type[Attributes]] = Attributes

    # Dataset mappings remain here until datasets become first-class models.
    _dataset_map: ClassVar[dict[str, str]] = {
        "Metadata": "metadata",
    }

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )

    attributes: Annotated[
        Attributes,
        Field(
            default_factory=Attributes,
            validation_alias=AliasChoices("attributes", "attrs"),
            serialization_alias="attrs",
        ),
    ]
    entity_type: Annotated[EntityType, Field(default_factory=EntityType)]
    parent_uid: UUID | None = None
    metadata: dict[str, Any] | None = Field(
        default=None,
        validation_alias=AliasChoices("metadata", "Metadata"),
        serialization_alias="Metadata",
    )
    on_file: bool = False

    @property
    def dataset_map(self) -> dict[str, str]:
        return self._dataset_map

    def h5_datasets(self) -> dict[str, Any]:
        """Serialize model fields that are stored as HDF5 datasets."""
        values = self.model_dump(
            include=set(self.dataset_map.values()),
            exclude_none=True,
        )
        return {
            h5_name: values[field_name]
            for h5_name, field_name in self.dataset_map.items()
            if field_name in values
        }

    @field_validator("metadata", mode="before")
    @classmethod
    def validate_metadata(cls, value: Any) -> dict[str, Any] | None:
        """Keep the common geoh5 metadata coercions local to the entity."""
        if isinstance(value, np.ndarray):
            value = value[0]

        if isinstance(value, bytes):
            value = value.decode("utf-8")

        if isinstance(value, str):
            # A future reader can replace this with format-aware JSON parsing.
            return {"raw": value}

        if value is not None and not isinstance(value, dict):
            raise ValueError(
                "Input metadata must be of type dict, bytes, str, ndarray or None."
            )

        return value

    @classmethod
    def from_legacy_entity(cls, entity: Any, **overrides) -> Self:
        """
        Build a Pydantic model from an existing geoh5py Entity-like object.

        This is a temporary migration adapter and is not used by the writer.
        """
        parent = getattr(entity, "parent", None)
        legacy_type = getattr(entity, "entity_type", None)
        type_values = {
            "uid": getattr(legacy_type, "uid", None),
            "name": getattr(legacy_type, "name", None),
            "description": getattr(legacy_type, "description", None),
        }
        attrs = {
            "entity_type": {
                key: value for key, value in type_values.items() if value is not None
            },
            "parent_uid": getattr(parent, "uid", None),
            "metadata": getattr(entity, "metadata", None),
            "on_file": getattr(entity, "on_file", False),
            "attributes": {
                key: getattr(entity, key, None)
                for key in Attributes.model_fields.keys()
                if getattr(entity, "uid", None) is not None
            },
        }
        attrs = {key: value for key, value in attrs.items() if value is not None}
        attrs.update(overrides)

        return cls.model_validate(attrs)

    def __setattr__(self, key, value):
        """
        Overload setattr method to deal with attributes.
        """

        if key in self.attributes.model_fields:
            setattr(self.attributes, key, value)
        else:
            setattr(self, key, value)

    def __getattr__(self, key):
        """
        Overload getattr method to deal with nested models.
        """
        if key in self.attributes.model_fields:
            return getattr(self.attributes, key)

        return getattr(self, key)
