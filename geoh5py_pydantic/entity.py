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

from collections.abc import Mapping
from typing import Any, ClassVar, Self, cast
from uuid import UUID

import numpy as np
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .entity_type import EntityType, NamedIdentity


def _field_input_names(field_name: str, field: Any) -> list[str]:
    """Return the Python name and accepted string validation aliases."""
    input_names = [field_name]
    if isinstance(field.validation_alias, AliasChoices):
        input_names.extend(
            choice
            for choice in field.validation_alias.choices
            if isinstance(choice, str)
        )
    elif isinstance(field.validation_alias, str):
        input_names.append(field.validation_alias)

    return input_names


def _merge_model_input(
    values: dict[str, Any],
    supplied: BaseModel | Mapping[str, Any],
    model_type: type[BaseModel],
) -> None:
    """Merge only explicitly supplied fields, normalized to Python names."""
    if isinstance(supplied, BaseModel):
        values.update(supplied.model_dump())
        return

    consumed = set()
    for field_name, field in model_type.model_fields.items():
        for input_name in _field_input_names(field_name, field):
            if input_name in supplied:
                values[field_name] = supplied[input_name]
                consumed.add(input_name)
                break

    # Preserve unknown nested keys so ``extra="forbid"`` reports them instead
    # of silently dropping values that cannot be written.
    values.update(
        {
            input_name: input_value
            for input_name, input_value in supplied.items()
            if input_name not in consumed
        }
    )


def _pop_model_input(
    values: dict[str, Any],
    supplied: dict[str, Any],
    model_type: type[BaseModel],
) -> None:
    """Move recognized model inputs out of a larger flat input dictionary."""
    for field_name, field in model_type.model_fields.items():
        for input_name in _field_input_names(field_name, field):
            if input_name in supplied:
                values[field_name] = supplied.pop(input_name)
                break


def _default_model_values(
    model_type: type[BaseModel],
    field_name: str,
    expected_type: type[BaseModel],
) -> tuple[dict[str, Any], BaseModel | None]:
    """Create the configured field default without confusing static linters."""
    model_fields = model_type.model_fields
    default = model_fields[field_name].get_default(call_default_factory=True)
    if isinstance(default, expected_type):
        return default.model_dump(), default

    return {}, None


def _pop_type_uid(values: dict[str, Any]) -> Any:
    """Remove legacy flat type UID aliases and return the first supplied value."""
    supplied_type_uid = None
    for input_name in (
        "type_uid",
        "Type ID",
        "Object Type ID",
        "Data Type ID",
    ):
        if input_name in values:
            candidate = values.pop(input_name)
            if supplied_type_uid is None:
                supplied_type_uid = candidate

    return supplied_type_uid


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
    last_focus: str = Field(
        default="None",
        validation_alias=AliasChoices("last_focus", "Last focus"),
        serialization_alias="Last focus",
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

    attributes: Attributes = Field(default_factory=Attributes)
    entity_type: EntityType = Field(default_factory=EntityType)
    parent_uid: UUID | None = None
    metadata: dict[str, Any] | None = Field(
        default=None,
        validation_alias=AliasChoices("metadata", "Metadata"),
        serialization_alias="Metadata",
    )
    on_file: bool = False

    @model_validator(mode="before")
    @classmethod
    def collect_flat_attributes(cls, value: Any) -> Any:
        """
        Accept the original flat constructor while storing values in models.

        This keeps ``PointsModel(name=..., allow_move=...)`` ergonomic and also
        accepts the explicit nested ``attributes=...`` and ``entity_type=...``
        forms introduced by the refactor.
        """
        if not isinstance(value, Mapping):
            return value

        values = dict(value)

        attribute_values, _ = _default_model_values(
            cls,
            "attributes",
            Attributes,
        )
        nested_attributes = values.get("attributes")
        if isinstance(nested_attributes, (Attributes | Mapping)):
            _merge_model_input(
                attribute_values,
                nested_attributes,
                cls.attributes_model,
            )

        _pop_model_input(attribute_values, values, cls.attributes_model)
        values["attributes"] = attribute_values

        supplied_type_uid = _pop_type_uid(values)
        supplied_entity_type = values.get("entity_type")
        if supplied_type_uid is not None or isinstance(
            supplied_entity_type, (EntityType | Mapping)
        ):
            type_values, default_entity_type = _default_model_values(
                cls,
                "entity_type",
                EntityType,
            )
            if isinstance(supplied_entity_type, (EntityType | Mapping)):
                entity_type_model = (
                    type(default_entity_type)
                    if isinstance(default_entity_type, EntityType)
                    else EntityType
                )
                _merge_model_input(
                    type_values,
                    supplied_entity_type,
                    entity_type_model,
                )
            if supplied_type_uid is not None:
                type_values["uid"] = supplied_type_uid
            values["entity_type"] = type_values

        return values

    def _attributes_value(self) -> Attributes:
        """Return the nested model with its runtime Pydantic type restored."""
        return cast(Attributes, self.attributes)

    def _entity_type_value(self) -> EntityType:
        """Return the nested type model with its runtime type restored."""
        return cast(EntityType, self.entity_type)

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

    @property
    def uid(self) -> UUID:
        return self._attributes_value().uid

    @uid.setter
    def uid(self, value: UUID) -> None:
        self._attributes_value().uid = value

    @property
    def name(self) -> str:
        return self._attributes_value().name

    @name.setter
    def name(self, value: str) -> None:
        self._attributes_value().name = value

    @property
    def type_uid(self) -> UUID:
        return self._entity_type_value().uid

    @type_uid.setter
    def type_uid(self, value: UUID) -> None:
        self._entity_type_value().uid = value

    @property
    def allow_delete(self) -> bool:
        return self._attributes_value().allow_delete

    @allow_delete.setter
    def allow_delete(self, value: bool) -> None:
        self._attributes_value().allow_delete = value

    @property
    def allow_move(self) -> bool:
        return self._attributes_value().allow_move

    @allow_move.setter
    def allow_move(self, value: bool) -> None:
        self._attributes_value().allow_move = value

    @property
    def allow_rename(self) -> bool:
        return self._attributes_value().allow_rename

    @allow_rename.setter
    def allow_rename(self, value: bool) -> None:
        self._attributes_value().allow_rename = value

    @property
    def clipping_ids(self) -> list[UUID] | None:
        return self._attributes_value().clipping_ids

    @clipping_ids.setter
    def clipping_ids(self, value: list[UUID] | None) -> None:
        self._attributes_value().clipping_ids = value

    @property
    def last_focus(self) -> str:
        return self._attributes_value().last_focus

    @last_focus.setter
    def last_focus(self, value: str) -> None:
        self._attributes_value().last_focus = value

    @property
    def partially_hidden(self) -> bool:
        return self._attributes_value().partially_hidden

    @partially_hidden.setter
    def partially_hidden(self, value: bool) -> None:
        self._attributes_value().partially_hidden = value

    @property
    def public(self) -> bool:
        return self._attributes_value().public

    @public.setter
    def public(self, value: bool) -> None:
        self._attributes_value().public = value

    @property
    def visible(self) -> bool:
        return self._attributes_value().visible

    @visible.setter
    def visible(self, value: bool) -> None:
        self._attributes_value().visible = value

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
            "uid": getattr(entity, "uid", None),
            "name": getattr(entity, "name", None),
            "entity_type": {
                key: value for key, value in type_values.items() if value is not None
            },
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
