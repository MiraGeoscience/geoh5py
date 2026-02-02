# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2020-2026 Mira Geoscience Ltd.                                '
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

import re
import uuid

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    model_serializer,
    model_validator,
)

from ..data import VisualParameters
from .object_base import ObjectBase


class PlatePosition(BaseModel):
    """
    Position of the top center of a Maxwell Plate.

    :param x: East coordinate.
    :param y: North coordinate.
    :param z: Elevation coordinate.
    :param increment: Increment value for the position.
    """

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    increment: float = Field(1.0, ge=1.0)

    parent: VisualParameters | None = Field(None, exclude=True)

    _initialized: bool = False

    @model_validator(mode="before")
    @classmethod
    def parse_str(cls, value: str | dict) -> dict:
        """
        Parse a string representation of the position stored
        as `inc; {x, y, z}`.

        :param value: String or dict representation of the position.

        :return: Dictionary of position parameters.
        """
        if isinstance(value, dict):
            return value

        coords = re.findall(r"[-+]?\d+\.\d+", value)
        args = dict(zip(["increment", "x", "y", "z"], coords, strict=False))
        return args

    @model_serializer(mode="plain")
    def format_to_str(self) -> str:
        """
        Format the position and increment to a string representation.

        :return: String representation of the position.
        """
        str_rep = "%.1f;{%.2f,%.2f,%.2f}" % tuple(
            getattr(self, key) for key in ["increment", "x", "y", "z"]
        )

        return str_rep

    @model_validator(mode="after")
    def validate_and_save(self):
        """
        Validate and save the position to the parent visual parameters.

        The initialization is deferred until a parent is assigned to
        prevent writing during model creation.
        """
        if not self._initialized:
            # Won't be initialized until a parent is assigned
            if self.parent is not None:
                self._initialized = True
            return self

        if self.parent is not None:
            # Model_dump configured to return a string representation
            self.parent.set_tags({"Position": self.model_dump()})  # pylint: disable=no-member

        return self


class PlateGeometry(BaseModel):
    """
    Geometry parameters for a Maxwell Plate.

    :param position: Top center coordinate of the plate with skew.
    :param dip: Dip angle of the plate in degrees.
    :param dip_direction: Dip direction of the plate in degrees.
    :param rotation: Rotation angle on the plane, about the top center coordinate.
    :param length: Length of the plate.
    :param width: Width of the plate.
    :param thickness: Thickness of the plate.
    :param number: Number of filaments to represent the plate.
    :param skew: Vertical deviation of the filament centers,
        as a ratio (>1: higher, 1: centered, [0, 1]: lower).
    """

    model_config = ConfigDict(
        validate_assignment=True, arbitrary_types_allowed=True, extra="ignore"
    )

    position: PlatePosition = Field(
        alias="Position", validation_alias=AliasChoices("Position", "position")
    )
    dip: float = Field(
        90.0, alias="Dip", validation_alias=AliasChoices("Dip", "dip"), ge=-90, le=90
    )
    dip_direction: float = Field(
        90.0,
        alias="Dipdirection",
        validation_alias=AliasChoices("Dipdirection", "dip_direction"),
        ge=0,
        le=360,
    )
    rotation: float = Field(
        0.0,
        alias="Rotation",
        validation_alias=AliasChoices("Rotation", "rotation"),
        ge=-180,
        le=180,
    )
    length: float = Field(
        1.0, alias="Length", validation_alias=AliasChoices("Length", "length"), ge=0
    )
    width: float = Field(
        1.0, alias="Width", validation_alias=AliasChoices("Width", "width"), ge=0
    )
    thickness: float = Field(
        1.0,
        alias="Thickness",
        validation_alias=AliasChoices("Thickness", "thickness"),
        ge=0,
    )
    number: int = Field(
        10,
        alias="Number",
        validation_alias=AliasChoices("Number", "number"),
        ge=0,
        le=50,
    )
    skew: float = Field(
        1.0,
        alias="Skew",
        validation_alias=AliasChoices("Skew", "skew"),
        ge=0.52,
        le=200.0,
    )

    parent: VisualParameters | None = Field(None, exclude=True)

    _initialized: bool = False

    @model_validator(mode="after")
    def validate_and_save(self):
        """
        Validate and save the geometry to the parent visual parameters.

        The initialization is deferred until a parent is assigned to
        prevent writing during model creation.
        """
        if not self._initialized:
            # Won't be initialized until a parent is assigned
            if self.parent is not None:
                self._initialized = True
                self.position.parent = self.parent
            return self

        if self.parent is not None:
            self.parent.set_tags(self.model_dump(by_alias=True))  # pylint: disable=no-member

        return self


class MaxwellPlate(ObjectBase):
    """
    Maxwell Plate object made up of visual parameters.
    """

    _default_name = "Maxwell Plate"
    _TYPE_UID: uuid.UUID | None = uuid.UUID("{878684e5-01bc-47f1-8c67-943b57d2e694}")

    def __init__(
        self,
        **kwargs,
    ):
        self._geometry: PlateGeometry | None = None
        super().__init__(**kwargs)

    @property
    def geometry(self) -> PlateGeometry | None:
        """
        Define the geometry of the Maxwell Plate.
        """
        if self._geometry is None and self.visual_parameters is not None:
            xml = self.visual_parameters.xml
            self._geometry = PlateGeometry(
                parent=self.visual_parameters,
                **{child.tag: child.text for child in xml},
            )

        return self._geometry

    @geometry.setter
    def geometry(self, value: PlateGeometry):
        """
        Set the geometry of the Maxwell Plate.

        :param value: Geometry parameters for the Maxwell Plate.
        """
        if not isinstance(value, PlateGeometry):
            raise TypeError("Input 'geometry' must be a PlateGeometry instance.")

        viz_params = self.visual_parameters
        if viz_params is None:
            viz_params = self.add_default_visual_parameters()

        value.parent = viz_params
        viz_params.set_tags(value.model_dump(by_alias=True))

        self._geometry = value

    @classmethod
    def create(
        cls, workspace, geometry: PlateGeometry | None = None, **kwargs
    ) -> MaxwellPlate:
        """
        Function to create an entity.

        :param workspace: Workspace to be added to.
        :param geometry: Geometry parameters for the Maxwell Plate.
        :param kwargs: List of keyword arguments defining the properties of a class.

        :return entity: Registered Entity to the workspace.
        """
        plate = super().create(workspace, **kwargs)

        if geometry is not None:
            plate.geometry = geometry

        return plate
