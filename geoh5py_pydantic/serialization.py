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

"""
A lot of this should probably be separated out into dedicated writer/utils files.

Serialize workspace-free Pydantic entities into an initialized geoh5 file.

The legacy H5Writer receives live entities and discovers their storage
location through inheritance and ``isinstance`` checks. This module instead
uses an explicit payload at least for now. The model owns validation and field serialization;
the writer owns only geoh5/HDF5 layout and encoding rules.

To make things easier, file initialization is intentionally outside this first writer.
A caller opens
an existing geoh5 file with h5py and retains ownership of that file handle.


serialization has two main stages:

for some model e.g. PointsModel:
- Geoh5EntityPayload: organize and serialize Python values
- Geoh5Writer: then, place those values into hdf5
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import h5py
import numpy as np
from pydantic import BaseModel

from .entity import PydanticEntity


# These child groups mirror the branches created by legacy
# H5Writer.write_entity. (Child groups that must be created inside each entity)
CHILD_COLLECTIONS = {
    "Data": (),
    "Groups": ("Data", "Groups", "Objects"),
    "Objects": ("Data",),
}

TYPE_COLLECTIONS = ("Data types", "Group types", "Object types")


@dataclass(slots=True)
class Geoh5EntityPayload:
    """
    All model information needed by the generic HDF5 writer.

    """

    # slots=True limits instances to these declared fields
    collection: str
    type_collection: str
    uid: UUID
    type_uid: UUID
    parent_uid: UUID | None
    attributes: dict[str, Any]
    datasets: dict[str, Any]
    type_attributes: dict[str, Any]

    @classmethod
    def from_model(cls, model: PydanticEntity) -> Geoh5EntityPayload:
        """Build a format-facing payload from a workspace-free model."""
        # read the model's placement declarations, raise error if one isn't provided.
        collection = model.geoh5_collection
        type_collection = model.geoh5_type_collection
        type_name = model.geoh5_type_name

        if collection is None:
            raise ValueError(f"{type(model).__name__} must declare 'geoh5_collection'.")

        if type_collection is None:
            raise ValueError(
                f"{type(model).__name__} must declare 'geoh5_type_collection'."
            )

        if model.type_uid is None:
            raise ValueError(
                f"{type(model).__name__} must have a type_uid to be saved."
            )

        if type_name is None:
            raise ValueError(f"{type(model).__name__} must declare 'geoh5_type_name'.")

        # EntityType uses Description, ID, and Name in the legacy writer. Have to supply
        # these here because no workspace-owned ObjectType exists.
        type_attributes = {
            "Description": model.geoh5_type_description or type_name,
            "ID": model.type_uid,
            "Name": type_name,
        }

        return cls(
            collection=collection,
            type_collection=type_collection,
            uid=model.uid,
            type_uid=model.type_uid,
            parent_uid=model.parent_uid,
            attributes=model.geoh5_attributes(),
            datasets=model.geoh5_datasets(),  # model field serialization occurs here
            type_attributes=type_attributes,
        )


class Geoh5Writer:
    """Write Pydantic entity payloads.

    For now, the supplied h5py file must already contain the core geoh5 project and Root
    structure. The writer does not register entities, mutate ``on_file``, or
    retain a reference on the model.
    """

    string_dtype = h5py.string_dtype()

    def __init__(self, h5file: h5py.File):
        if not isinstance(h5file, h5py.File):
            raise TypeError("Geoh5Writer requires an open h5py.File right now.")

        project_names = list(h5file)  # probably "GEOSCIENCE"
        if len(project_names) != 1:
            raise ValueError(
                "A geoh5 file must contain exactly one top-level project group."
            )

        self.h5file = h5file
        self.project = h5file[project_names[0]]
        # can't create a blank .geoh5 from scratch. validate the existing one contains
        # everything we expect e.g. Data, Groups, Objects, Root, Types, etc
        self._validate_project_structure()

    def write(
        self,
        model: PydanticEntity,
        *,
        compression: int = 5,
    ) -> h5py.Group:
        """Create a payload from ``model`` and write it to the open file."""
        return self.write_payload(
            Geoh5EntityPayload.from_model(model), compression=compression
        )

    def write_payload(
        self,
        payload: Geoh5EntityPayload,
        *,
        compression: int = 5,
    ) -> h5py.Group:
        """
        Write one generic entity payload and return its canonical group.
        This is what's performing the actual HDF5 change.
        """
        self._validate_payload(payload, compression)

        uid = self.format_uuid(payload.uid)
        collection_group = self.project[payload.collection]
        if uid in collection_group:
            # the writer currently does creation only, so raise an error if the uid exists already
            raise FileExistsError(
                f"Entity {payload.uid} already exists in {payload.collection}."
            )

        # Resolve the parent and type before creating the entity
        parent_group = self._find_parent(payload.parent_uid)
        type_group = self._ensure_type(payload)

        # Create the entity group under its collection
        # e.g. for Points, that becomes "/GEOSCIENCE/Objects/{points_uid}"
        entity_group = collection_group.create_group(uid, track_order=True)

        try:
            # Create child collection groups, write scalar attributes, write datasets,
            # link the entity to its type, link the entity from its parent

            for child_collection in CHILD_COLLECTIONS[payload.collection]:
                entity_group.create_group(child_collection, track_order=True)

            self._write_attributes(entity_group, payload.attributes)
            self._write_datasets(entity_group, payload.datasets, compression)

            # Both assignments create HDF5 hard links. Match
            # H5Writer.write_entity and H5Writer.write_to_parent.
            #
            # The same object can be reached through
            # GEOSCIENCE/Objects/{uid}  or GEOSCIENCE/Root/Objects/{uid}
            entity_group["Type"] = type_group
            parent_group.require_group(payload.collection)[uid] = entity_group

        except Exception:
            # Don't leave a partially written canonical entity behind.
            del collection_group[uid]
            raise

        return entity_group

    def _validate_project_structure(self) -> None:
        """Fail early when the handle is HDF5 but not an initialized geoh5."""
        required = {*CHILD_COLLECTIONS, "Root", "Types"}
        missing = sorted(required.difference(self.project))
        if missing:
            raise ValueError(
                "The HDF5 file is missing geoh5 project groups: " + ", ".join(missing)
            )

        type_root = self.project["Types"]
        missing_types = sorted(set(TYPE_COLLECTIONS).difference(type_root))
        if missing_types:
            raise ValueError(
                "The geoh5 Types group is missing: " + ", ".join(missing_types)
            )

    def _validate_payload(self, payload: Geoh5EntityPayload, compression: int) -> None:
        if payload.collection not in CHILD_COLLECTIONS:
            raise ValueError(f"Unsupported geoh5 collection '{payload.collection}'.")

        if payload.type_collection not in TYPE_COLLECTIONS:
            raise ValueError(
                f"Unsupported geoh5 type collection '{payload.type_collection}'."
            )

        if isinstance(compression, bool) or not isinstance(compression, int):
            raise TypeError("Compression must be an integer from 0 to 9.")

        if not 0 <= compression <= 9:
            raise ValueError("Compression must be between 0 and 9.")

    def _ensure_type(self, payload: Geoh5EntityPayload) -> h5py.Group:
        """
        Create or reuse the shared EntityType group for this model.
        """
        types = self.project["Types"][payload.type_collection]
        type_uid = self.format_uuid(payload.type_uid)

        #
        if type_uid in types:
            return types[type_uid]

        type_group = types.create_group(type_uid, track_order=True)
        self._write_attributes(type_group, payload.type_attributes)
        return type_group

    def _find_parent(self, parent_uid: UUID | None) -> h5py.Group:
        """Resolve a parent UID from the file, defaulting to the project Root."""
        root = self.project["Root"]
        if parent_uid is None:
            return root

        parent_name = self.format_uuid(parent_uid)
        if self._attribute_as_text(root.attrs.get("ID")) == parent_name:
            return root

        # Entities capable of containing children currently live in Groups or
        # Objects. This replaces legacy ``entity.parent`` object traversal.
        for collection in ("Groups", "Objects"):
            if parent_name in self.project[collection]:
                return self.project[collection][parent_name]

        raise KeyError(f"Parent {parent_uid} does not exist in the geoh5 file.")

    def _write_attributes(
        self, group: h5py.Group, attributes: Mapping[str, Any]
    ) -> None:
        for name, value in attributes.items():
            if value is not None:
                self._write_attribute(group, name, value)

    def _write_attribute(self, group: h5py.Group, name: str, value: Any) -> None:
        """Apply the scalar encoding rules from legacy create_attribute."""
        if isinstance(value, UUID):
            value = self.format_uuid(value)

        if isinstance(value, (bool, np.bool_)):
            group.attrs.create(name, int(value), dtype="int8")
            return

        if isinstance(value, str):
            group.attrs.create(name, value, dtype=self.string_dtype)
            return

        if isinstance(value, BaseModel):
            group.attrs.create(
                name,
                value.model_dump_json(by_alias=True),
                dtype=self.string_dtype,
            )
            return

        text_values = self._text_sequence(value)
        if text_values is not None:
            group.attrs.create(
                name,
                np.asarray(text_values, dtype=object),
                dtype=self.string_dtype,
            )
            return

        array = np.asarray(value)
        if array.dtype.kind == "O":
            raise TypeError(
                f"Attribute '{name}' has unsupported value type {type(value)}."
            )

        group.attrs.create(name, value, dtype=array.dtype)

    def _write_datasets(
        self,
        group: h5py.Group,
        datasets: Mapping[str, Any],
        compression: int,
    ) -> None:
        """
        Dispatch by value type.
        """
        for name, value in datasets.items():
            if isinstance(value, np.ndarray):
                self._write_array_dataset(group, name, value, compression)
            elif isinstance(value, (Mapping | BaseModel)):
                self._write_json_dataset(group, name, value)
            elif isinstance(value, str):
                group.create_dataset(
                    name,
                    data=np.asarray([value], dtype=object),
                    dtype=self.string_dtype,
                )
            elif value is not None:
                raise TypeError(
                    f"Dataset '{name}' has unsupported value type {type(value)}."
                )

    def _write_array_dataset(
        self,
        group: h5py.Group,
        name: str,
        values: np.ndarray,
        compression: int,
    ) -> None:
        """Write arrays using the behavior of write_array_attribute."""
        if np.issubdtype(values.dtype, np.str_):
            values = values.astype(self.string_dtype)

        options: dict[str, Any] = {}
        if values.ndim > 0:
            options = {
                "compression": "gzip",
                "compression_opts": compression,
            }

        group.create_dataset(name, data=values, **options)

    def _write_json_dataset(
        self,
        group: h5py.Group,
        name: str,
        value: Mapping[str, Any] | BaseModel,
    ) -> None:
        """Write JSON text in the one-element layout used by write_metadata."""
        # prepare the value, then turn the prepared value into json text
        json_text = json.dumps(self._json_value(value), indent=4)

        group.create_dataset(
            name,
            data=np.asarray([json_text], dtype=object),
            dtype=self.string_dtype,
        )

    @classmethod
    def _json_value(cls, value: Any) -> Any:  # pylint: disable=too-many-return-statements
        """
        Recursively convert UUID and NumPy values before JSON encoding. UUIDs become braced strings,
        Pydantic models become dicts, nested dicts and sequences are recursively converted to json
        values, numpy arrays become lists, numpy scalars become normal python scalars.

        """
        if isinstance(value, UUID):
            # UUID can't be written directly, so turn it to a string
            return cls.format_uuid(value)

        if isinstance(value, BaseModel):
            # turn the model into a dictionary, then recursively convert its values to json values
            return cls._json_value(value.model_dump(mode="python", by_alias=True))

        if isinstance(value, Mapping):
            # for dicts and other mappings, loop over every k/v pair and convert both
            return {
                str(cls._json_value(key)): cls._json_value(item)
                for key, item in value.items()
            }

        if isinstance(value, Sequence) and not isinstance(value, (str | bytes)):
            # lists, tuples, sequences are converted to lists of json values
            return [cls._json_value(item) for item in value]

        if isinstance(value, np.ndarray):
            # for numpy arrays, turn to normal list so json.dumps() works on them
            return cls._json_value(value.tolist())

        if isinstance(value, np.generic):
            # for numpy scalar values, e.g. np.float32, np.bool_, convert to normal python scalar
            # which is all that's needed to be handled by json
            return value.item()

        # normal json-compatible values don't need to be converted
        return value

    @classmethod
    def _text_sequence(cls, value: Any) -> list[str] | None:
        """Format UUID/string attribute sequences, notably Clipping IDs."""
        if not isinstance(value, Sequence) or isinstance(value, (str | bytes)):
            return None

        values = list(value)
        if not values:
            return []

        if not all(isinstance(item, (str | UUID)) for item in values):
            return None

        return [
            cls.format_uuid(item) if isinstance(item, UUID) else item for item in values
        ]

    @staticmethod
    def _attribute_as_text(value: Any) -> str | None:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, str):
            return value
        return None

    @staticmethod
    def format_uuid(value: UUID) -> str:
        """Mirrors geoh5py.shared.utils.as_str_if_uuid."""
        return "{" + str(value) + "}"
