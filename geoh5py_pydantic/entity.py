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
from typing import Annotated, Any, ClassVar, Self, cast
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


# used to distinguish "not supplied" from "explicitly supplied as None" for certain fields where
# None is valid
_MISSING = object()

_TYPE_UID_INPUT_NAMES = (
    "type_uid",
    "Type ID",
    "Object Type ID",
    "Data Type ID",
)


def _field_input_names(field_name: str, field: Any) -> list[str]:
    """Return a field's Python name and accepted string aliases."""
    input_names = [field_name]
    validation_alias = field.validation_alias

    # Add the validation alias(es) to the list of input names
    if isinstance(validation_alias, AliasChoices):
        input_names.extend(
            alias for alias in validation_alias.choices if isinstance(alias, str)
        )
    elif isinstance(validation_alias, str):
        input_names.append(validation_alias)

    return input_names


def _model_field_names(model_type: type[BaseModel]) -> list[str]:
    """
    Return model field names behind a Pydantic-aware type boundary.
    Mainly used as a type boundary for Pylint, which may otherwise report Pydantic's
    model_fields as non-iterable
    """
    return list(model_type.model_fields)


def _default_nested_values(
    model_type: type[BaseModel],
    field_name: str,
    expected_type: type[BaseModel],
) -> dict[str, Any]:
    """Return a nested model's configured defaults as ordinary values."""
    field = model_type.model_fields.get(field_name)
    if field is None:
        return {}

    default = field.get_default(call_default_factory=True)
    if isinstance(default, expected_type):
        return default.model_dump()

    return {}


def _normalize_nested_input(
    supplied: BaseModel | Mapping[str, Any],
    model_type: type[BaseModel],
) -> dict[str, Any]:
    """Normalize known aliases while preserving unknown keys for validation."""
    if isinstance(supplied, BaseModel):
        return supplied.model_dump()

    normalized: dict[str, Any] = {}
    consumed: set[str] = set()
    for field_name, field in model_type.model_fields.items():
        for input_name in _field_input_names(field_name, field):
            if input_name in supplied:
                normalized[field_name] = supplied[input_name]
                consumed.add(input_name)
                break

    normalized.update(
        {
            input_name: input_value
            for input_name, input_value in supplied.items()
            if input_name not in consumed
        }
    )
    return normalized


def _pop_flat_model_input(
    values: dict[str, Any],
    model_type: type[BaseModel],
) -> dict[str, Any]:
    """Move fields accepted by a nested model out of a flat input mapping."""
    nested_values = {}
    for field_name, field in model_type.model_fields.items():
        for input_name in _field_input_names(field_name, field):
            if input_name in values:
                nested_values[field_name] = values.pop(input_name)
                break

    return nested_values


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
        default=None,
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
        extra="forbid",  # prevent unwritable values from being silently discarded
        populate_by_name=True,  # permit Python field names alongside aliases
        validate_assignment=True,  # validate changes after creation
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

    @classmethod
    def _collect_attribute_input(cls, values: dict[str, Any]) -> None:
        """Route flat and nested inputs into the attributes model."""
        nested_attributes = _MISSING
        if "attributes" in values:
            nested_attributes = values.pop("attributes")
        elif "attrs" in values:
            nested_attributes = values.pop("attrs")

        flat_attribute_values = _pop_flat_model_input(
            values,
            cls.attributes_model,
        )
        valid_nested_attributes = isinstance(
            nested_attributes,
            (BaseModel, Mapping),
        )
        if (
            isinstance(nested_attributes, cls.attributes_model)
            and not flat_attribute_values
        ):
            # Assignment validation supplies existing fields to this validator.
            # Keep an already-valid model intact when another field is changing.
            values["attributes"] = nested_attributes
        elif nested_attributes is _MISSING or valid_nested_attributes:
            attribute_values = _default_nested_values(
                cls,
                "attributes",
                cls.attributes_model,
            )
            if valid_nested_attributes:
                attribute_values.update(
                    _normalize_nested_input(
                        cast(
                            BaseModel | Mapping[str, Any],
                            nested_attributes,
                        ),
                        cls.attributes_model,
                    )
                )
            attribute_values.update(flat_attribute_values)
            values["attributes"] = attribute_values
        else:
            # Preserve invalid input so Pydantic reports it at ``attributes``.
            values["attributes"] = nested_attributes

    @classmethod
    def _collect_entity_type_input(cls, values: dict[str, Any]) -> None:
        """Route a nested type and flat type UID into the entity type model."""
        supplied_entity_type = values.pop("entity_type", _MISSING)
        supplied_type_uid = _MISSING
        for input_name in _TYPE_UID_INPUT_NAMES:
            if input_name in values:
                supplied_type_uid = values.pop(input_name)
                break

        if (
            isinstance(supplied_entity_type, EntityType)
            and supplied_type_uid is _MISSING
        ):
            # As above, avoid replacing an untouched model during assignment.
            values["entity_type"] = supplied_entity_type
        elif supplied_entity_type is not _MISSING or supplied_type_uid is not _MISSING:
            type_values = _default_nested_values(
                cls,
                "entity_type",
                EntityType,
            )
            valid_entity_type = isinstance(
                supplied_entity_type,
                (BaseModel, Mapping),
            )
            if valid_entity_type:
                type_values.update(
                    _normalize_nested_input(
                        supplied_entity_type,
                        EntityType,
                    )
                )
            elif supplied_entity_type is not _MISSING:
                values["entity_type"] = supplied_entity_type

            if supplied_entity_type is _MISSING or valid_entity_type:
                if supplied_type_uid is not _MISSING:
                    type_values["uid"] = supplied_type_uid
                values["entity_type"] = type_values

    @model_validator(mode="before")
    @classmethod
    def collect_flat_model_fields(cls, value: Any) -> Any:
        """
        Route convenient flat inputs into the nested models that own them.

        The instance-level forwarding methods apply only after construction.
        This adapter lets ``PointsModel(name=...)`` and geoh5 aliases such as
        ``Name`` retain that same flat API during Pydantic validation.
        """
        if not isinstance(value, Mapping):
            return value

        values = dict(value)
        cls._collect_attribute_input(values)
        cls._collect_entity_type_input(values)

        return values

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
        attribute_values = {}
        for field_name in _model_field_names(cls.attributes_model):
            field_value = getattr(entity, field_name, _MISSING)
            if field_value is not _MISSING:
                attribute_values[field_name] = field_value

        attrs = {
            "entity_type": {
                key: value for key, value in type_values.items() if value is not None
            },
            "parent_uid": getattr(parent, "uid", None),
            "metadata": getattr(entity, "metadata", None),
            "on_file": getattr(entity, "on_file", False),
            "attributes": attribute_values,
        }
        attrs = {key: value for key, value in attrs.items() if value is not None}
        attrs.update(overrides)

        return cls.model_validate(attrs)

    @property
    def type_uid(self) -> UUID:
        """UID of the shared entity type."""
        return self.entity_type.uid

    @type_uid.setter
    def type_uid(self, value: UUID) -> None:
        self.entity_type.uid = value

    def __setattr__(self, key: str, value: Any) -> None:
        """Forward core attribute assignment to the nested attributes model."""
        attributes = self.__dict__.get("attributes")
        if isinstance(attributes, BaseModel) and key in type(attributes).model_fields:
            setattr(attributes, key, value)
            return

        super().__setattr__(key, value)

    def __getattr__(self, key: str) -> Any:
        """Forward core attribute access to the nested attributes model."""
        attributes = self.__dict__.get("attributes")
        if isinstance(attributes, BaseModel) and key in type(attributes).model_fields:
            return getattr(attributes, key)

        # Pydantic defines BaseModel.__getattr__ only at runtime.
        return super().__getattr__(key)  # type: ignore[misc]
