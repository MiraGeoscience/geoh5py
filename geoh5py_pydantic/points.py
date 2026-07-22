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
from typing import Any, ClassVar, Self
from uuid import UUID

import numpy as np
from pydantic import AliasChoices, Field, field_serializer, field_validator

from .arrays import ArraySource, LazyArray
from .entity import PydanticEntity


POINTS_TYPE_UID = UUID("{202C5DB1-A56D-4004-9CAD-BAAFD8899406}")
VERTICES_DTYPE = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8")])


def _coerce_vertices_dtype(value: Any) -> Any:
    """
    Normalize a structured ``VERTICES_DTYPE`` array into a plain ``(n, 3)``
    float array. Anything else is passed through unchanged.

    This is attached as a ``LazyArray`` validator so the coercion runs
    consistently whether vertices are loaded lazily or supplied eagerly.
    """
    if isinstance(value, np.ndarray) and value.dtype == VERTICES_DTYPE:
        return value.view("<f8").reshape((-1, 3))

    return value


def _vertices_to_geoh5(value: np.ndarray) -> np.ndarray:
    """
    Convert a plain ``(n, 3)`` float array into the structured dtype used
    for vertices in geoh5 files.

    Attached as the ``LazyArray`` serializer so vertices can serialize
    themselves in-chain via :meth:`LazyArray.to_geoh5`.
    """
    return np.asarray(np.rec.fromarrays(value.T.tolist(), dtype=VERTICES_DTYPE))


class PointsModel(PydanticEntity):
    """
    Experimental pydantic model for a geoh5 Points object.

    The model can be instantiated without a workspace. Large arrays may be
    supplied directly or as :class:`LazyArray` instances backed by an adapter.
    """

    # The legacy writer derives these locations from isinstance(ObjectBase) and
    # isinstance(ObjectType); declaring them on the model removes that coupling.
    geoh5_collection: ClassVar[str] = "Objects"
    geoh5_type_collection: ClassVar[str] = "Object types"

    # Legacy ObjectType metadata can depend on the first entity written. A
    # stable class-level name makes the file independent of entity write order.
    geoh5_type_name: ClassVar[str] = "Points"
    geoh5_type_description: ClassVar[str] = "Points"

    # For attribute and dataset maps, copy base entity map and add object-specific attributes
    _attribute_map: ClassVar[dict[str, str]] = {
        **PydanticEntity._attribute_map,
        "Last focus": "last_focus",
    }

    _dataset_map: ClassVar[dict[str, str]] = {
        **PydanticEntity._dataset_map,
        "Vertices": "vertices",
    }

    type_uid: UUID = Field(
        default=POINTS_TYPE_UID,
        validation_alias=AliasChoices(
            "type_uid", "Type ID", "Object Type ID", "Data Type ID"
        ),
        serialization_alias="Object Type ID",
    )
    name: str = Field(
        default="Points",
        validation_alias=AliasChoices("name", "Name"),
        serialization_alias="Name",
    )
    last_focus: str = Field(
        default="None",
        validation_alias=AliasChoices("last_focus", "Last focus"),
        serialization_alias="Last focus",
    )
    vertices: LazyArray = Field(
        default=None,
        validate_default=True,
        validation_alias=AliasChoices("vertices", "Vertices"),
        serialization_alias="Vertices",
    )

    minimum_vertices: ClassVar[int] = 1

    @field_validator("vertices", mode="before")
    @classmethod
    def validate_vertices_field(cls, value: Any) -> LazyArray:
        if isinstance(value, LazyArray):
            return (
                value.with_validator(_coerce_vertices_dtype)
                .with_validator(cls.validate_vertices)
                .with_serializer(_vertices_to_geoh5)
            )

        return LazyArray(
            source=None,
            uid=None,
            key="vertices",
            validator=[_coerce_vertices_dtype, cls.validate_vertices],
            serializer=_vertices_to_geoh5,
            value=value,
        )

    @classmethod
    def validate_vertices(cls, value: Any) -> np.ndarray:
        """
        Validate vertices as a plain ``(n, 3)`` float array.

        The legacy ``Points`` object stores vertices as a structured array on
        disk. This model keeps a friendlier numpy representation internally;
        the structured dtype used at IO boundaries is handled by the
        ``LazyArray`` itself (see ``_coerce_vertices_dtype``/
        ``_vertices_to_geoh5``).
        """
        if value is None:
            warnings.warn(
                "No 'vertices' provided. Using (0, 0, 0) default point at the origin.",
                UserWarning,
            )
            value = (0.0, 0.0, 0.0)

        if isinstance(value, (list, tuple)):
            value = np.array(value, ndmin=2)

        if not isinstance(value, np.ndarray):
            raise ValueError(
                "Vertices must be a numpy array, list, tuple or LazyArray."
            )

        if len(value) < cls.minimum_vertices:
            warnings.warn(
                f"Attribute 'vertices' has fewer elements than the minimum required "
                f"for object of type {cls}. Augmenting the array to shape "
                f"({cls.minimum_vertices}, 3).",
                UserWarning,
            )
            value = np.vstack([value] * cls.minimum_vertices)

        if not np.issubdtype(value.dtype, np.number):
            raise ValueError(f"Array of 'vertices' must be numeric. Got {value.dtype}.")

        if value.ndim != 2 or value.shape[-1] != 3:
            raise ValueError(
                "Array of 'vertices' should be of shape (*, 3). "
                f"Got shape {value.shape}."
            )

        return np.asarray(value, dtype="<f8")

    @field_serializer("vertices")
    def _serialize_vertices(self, value: LazyArray) -> np.ndarray:
        """
        Serialize vertices using the geoh5 structured dtype, delegating the
        conversion to the ``LazyArray`` itself via ``to_geoh5``.
        """
        return value.to_geoh5()

    @property
    def vertices_array(self) -> np.ndarray:
        """
        Vertices as an eager numpy array.
        """
        return np.asarray(self.vertices)

    @property
    def n_vertices(self) -> int:
        """
        Number of vertices.
        """
        return self.vertices_array.shape[0]

    @property
    def locations(self) -> np.ndarray:
        """
        Coordinate locations represented by this object.
        """
        return self.vertices_array

    @property
    def extent(self) -> np.ndarray:
        """
        Bounding box of vertices, shaped ``(2, 3)``.
        """
        vertices = self.vertices_array
        return np.c_[vertices.min(axis=0), vertices.max(axis=0)].T

    def as_geoh5_vertices(self) -> np.ndarray:
        """
        Convert vertices to the structured dtype used in geoh5 files.

        Delegates to :meth:`LazyArray.to_geoh5`, which owns the conversion
        via the ``_vertices_to_geoh5`` serializer attached to the field.
        """
        return self.vertices.to_geoh5()  # pylint: disable=no-member

    @classmethod
    def from_array_source(
        cls,
        source: ArraySource,
        uid: UUID,
        *,
        key: str = "vertices",
        **attributes,
    ) -> Self:
        """
        Build a PointsModel with vertices loaded lazily from an array source.
        """
        return cls(
            uid=uid,
            vertices=LazyArray(source, uid, key, validator=[]),
            **attributes,
        )

    @classmethod
    def from_legacy_points(cls, points: Any, **overrides) -> Self:
        """
        Build a pydantic model from the current geoh5py Points object.

        This adapter is eager for now. A later geoh5 file adapter should provide
        true lazy loading without going through ``Workspace.load_entity``.
        """
        return cls.from_legacy_entity(
            points,
            vertices=getattr(points, "vertices", None),
            last_focus=getattr(points, "last_focus", "None"),
            **overrides,
        )

    def model_dump_geoh5_attributes(self) -> dict[str, Any]:
        """
        Dump the geoh5 attributes for this points object.
        """
        return self.geoh5_attributes()

    def model_dump_everything(self) -> dict[str, Any]:
        """
        Method just for example cases where being able to dump everything like a previous version
        of the model_dump_geoh5_attributes method would be useful for notebook display

        """
        return {
            "attributes": self.geoh5_attributes(),
            "datasets": self.geoh5_datasets(),
            "type_uid": self.type_uid,
            "parent_uid": self.parent_uid,
        }
