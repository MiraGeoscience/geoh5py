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
from pydantic import AliasChoices, Field, field_validator

from .arrays import ArraySource, LazyArray
from .entity import PydanticEntity


POINTS_TYPE_UID = UUID("{202C5DB1-A56D-4004-9CAD-BAAFD8899406}")
VERTICES_DTYPE = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8")])


class PointsModel(PydanticEntity):
    """
    Experimental pydantic model for a geoh5 Points object.

    The model can be instantiated without a workspace. Large arrays may be
    supplied directly or as :class:`LazyArray` instances backed by an adapter.
    """

    type_uid: UUID = POINTS_TYPE_UID
    name: str = "Points"
    last_focus: str = Field(
        default="None", validation_alias=AliasChoices("last_focus", "Last focus")
    )
    vertices: np.ndarray | LazyArray = Field(
        default=None,
        validate_default=True,
        validation_alias=AliasChoices("vertices", "Vertices"),
    )

    minimum_vertices: ClassVar[int] = 1

    @field_validator("vertices", mode="before")
    @classmethod
    def validate_vertices_field(cls, value: Any) -> np.ndarray | LazyArray:
        if isinstance(value, LazyArray):
            return value.with_validator(cls.validate_vertices)

        return cls.validate_vertices(value)

    @classmethod
    def validate_vertices(cls, value: Any) -> np.ndarray:
        """
        Validate vertices as a plain ``(n, 3)`` float array.

        The legacy ``Points`` object stores vertices as a structured array on
        disk. This model keeps a friendlier numpy representation internally and
        converts to the structured dtype only at IO boundaries.
        """
        if value is None:
            warnings.warn(
                "No 'vertices' provided. Using (0, 0, 0) default point at the origin.",
                UserWarning,
            )
            value = (0.0, 0.0, 0.0)

        if isinstance(value, LazyArray):
            return value.load()

        if isinstance(value, (list, tuple)):
            value = np.array(value, ndmin=2)

        if not isinstance(value, np.ndarray):
            raise ValueError(
                "Vertices must be a numpy array, list, tuple or LazyArray."
            )

        if value.dtype == VERTICES_DTYPE:
            value = value.view("<f8").reshape((-1, 3))

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
        """
        return np.asarray(
            np.rec.fromarrays(self.vertices_array.T.tolist(), dtype=VERTICES_DTYPE)
        )

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
            vertices=LazyArray(source, uid, key, validator=cls.validate_vertices),
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
        attrs = super().model_dump_geoh5_attributes()
        attrs.update(
            {
                "Object Type ID": self.type_uid,
                "Last focus": self.last_focus,
                "Vertices": self.as_geoh5_vertices(),
            }
        )
        return attrs
