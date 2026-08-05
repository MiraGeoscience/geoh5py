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

import warnings
from collections.abc import Mapping
from typing import (  # type: ignore[attr-defined]
    Annotated,
    Any,
    ClassVar,
    GenericAlias,
    Self,
)
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


def collect_input_from_dict(
    model: type[BaseModel], data: dict[str, Any]
) -> dict[str, dict | Any]:
    """
    Recursively replace BaseModel objects with nested dictionary of 'data' values.

    :param base_model: BaseModel object to structure data for.
    :param data: Flat dictionary of parameters and values without nesting structure.
    """
    update = data.copy()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

        for field, info in model.model_fields.items():
            # Already a BaseModel, no need to nest
            if isinstance(update.get(field, None), BaseModel):
                continue

            if (
                isinstance(info.annotation, type)
                and not isinstance(info.annotation, GenericAlias)
                and issubclass(info.annotation, BaseModel)
            ):
                # Nest and deal with aliases
                update = collect_input_from_dict(info.annotation, update)
                nested = info.annotation.model_construct(**update).model_dump(
                    exclude_unset=True
                )
                aliases = info.annotation.model_construct(**update).model_dump(
                    exclude_unset=True, by_alias=True
                )
                if any(nested):
                    update[field] = nested

                    for key, alias in zip(nested, aliases, strict=True):
                        if key in update:
                            del update[key]
                        if alias in update:
                            del update[alias]

    return update


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
        values = collect_input_from_dict(cls, values)

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
        for field_name in list(cls.attributes_model.model_fields.keys()):
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
