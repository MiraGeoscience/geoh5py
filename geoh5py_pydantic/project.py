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

"""Validated attributes stored on the top-level geoh5 project group."""

from __future__ import annotations

from getpass import getuser
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


def _default_contributors() -> tuple[str, ...]:
    """Evaluate the current user when a project is created, not at import time."""
    return (getuser(),)


class ProjectAttributes(BaseModel):
    """
    Attributes stored on the top-level HDF5 project group.

    Python field names remain convenient in model code. Serialization aliases
    are the exact attribute names stored in a geoh5 file.
    """

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )

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

    @field_validator("contributors")
    @classmethod
    def validate_contributors(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not contributor for contributor in value):
            raise ValueError("Contributors must contain at least one non-empty name.")
        return value

    def h5_attributes(self) -> dict[str, Any]:
        """Return project values using their exact geoh5 attribute names."""
        return self.model_dump(by_alias=True)
