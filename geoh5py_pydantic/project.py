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

"""Validated project-level values used when creating a geoh5 file."""

from __future__ import annotations

from getpass import getuser
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


DEFAULT_PAGE_SIZE = 65_536  # Same as legacy workspace default


def _default_contributors() -> tuple[str, ...]:
    """Evaluate the current user when a project is created, not at import time."""
    return (getuser(),)


class Geoh5Project(BaseModel):
    """
    File-level settings and attributes for a geoh5 project.

    ``name`` identifies the top-level HDF5 group and ``page_size`` configures
    the HDF5 file itself. The remaining fields become attributes on the project
    group and therefore use their exact geoh5 names as serialization aliases.
    """

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )

    name: str = "GEOSCIENCE"
    contributors: tuple[str, ...] = Field(
        default_factory=_default_contributors,
        validation_alias=AliasChoices("contributors", "Contributors"),
        serialization_alias="Contributors",
    )
    distance_unit: str = Field(
        default="meter",
        validation_alias=AliasChoices("distance_unit", "Distance unit"),
        serialization_alias="Distance unit",
    )
    ga_version: str = Field(
        default="1",
        validation_alias=AliasChoices("ga_version", "GA Version"),
        serialization_alias="GA Version",
    )
    version: float = Field(
        default=2.1,
        validation_alias=AliasChoices("version", "Version"),
        serialization_alias="Version",
    )
    # strict=True ensures that the value is an int
    # and not a float, string, or other type that gets coerced to an int
    page_size: int = Field(default=DEFAULT_PAGE_SIZE, strict=True)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Require one top-level HDF5 group rather than a nested path."""
        if not value or "/" in value:
            raise ValueError("Project name must be non-empty and cannot contain '/'.")
        return value

    @field_validator("contributors")
    @classmethod
    def validate_contributors(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not contributor for contributor in value):
            raise ValueError("Contributors must contain at least one non-empty name.")
        return value

    @field_validator("page_size")
    @classmethod
    def validate_page_size(cls, value: int) -> int:
        """Mirror the legacy Workspace page-size requirements."""
        if value < 512 or value % 2 != 0:
            raise ValueError("Page size must be a multiple of 2 and at least 512.")
        return value

    def h5_attributes(self) -> dict[str, Any]:
        """
        Return only values stored as attributes on the project group.
        (i.e., return values written onto ``/GEOSCIENCE`` rather than the HDF5 file itself).
        """
        return self.model_dump(
            by_alias=True,
            exclude={"name", "page_size"},
        )

    def h5_creation_options(self) -> dict[str, Any]:
        """
        Return the HDF5 options used by the legacy Workspace creator.
        (i.e., return options passed to ``h5py.File`` -- ``name`` and ``page_size`` are not project
        attributes because one names the group and the other configures the physical file.)
        """
        return {
            "fs_strategy": "page",
            "page_buf_size": self.page_size * 256,
            "fs_page_size": self.page_size,
            "libver": ("v110", "v114"),
        }
