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
from contextvars import ContextVar
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    ValidationInfo,
    field_validator,
    model_serializer,
)

from ..data import VisualParameters
from .object_base import ObjectBase


CONTEXT_PARENT: ContextVar[VisualParameters] = ContextVar("parent")


class PlatePosition(BaseModel):
    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    increment: float = 0.0

    parent: VisualParameters | None = Field(None, exclude=True)

    @field_validator("*", mode="after")
    @classmethod
    def save_to_visual_parameters(cls, value: str, info: ValidationInfo):
        if (
            len(cls.model_fields.keys()) != len(info.data) + 1
            or info.field_name == "parent"
        ):
            return value

        element = CONTEXT_PARENT.get().xml.find("Position")

        if element is not None:
            element.text = "%.1f;{%.2f,%.2f,%.2f}" % tuple(
                info.data.get(key, value) for key in ["increment", "x", "y", "z"]
            )
            CONTEXT_PARENT.get().workspace.update_attribute(
                CONTEXT_PARENT.get(), "values"
            )

        return value


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

    position: PlatePosition = Field(alias="Position")
    dip: float = Field(90.0, alias="Dip")
    dip_direction: float = Field(90.0, alias="Dipdirection")
    rotation: float = Field(0.0, alias="Rotation")
    length: float = Field(1.0, alias="Length")
    width: float = Field(1.0, alias="Width")
    thickness: float = Field(1.0, alias="Thickness")
    number: int = Field(10, alias="Number")
    skew: float = Field(1.0, alias="Skew")

    parent: VisualParameters = Field(exclude=True)

    @field_validator("position", mode="before")
    @classmethod
    def parse_position(cls, value: str):
        args: dict[str, Any] = {"increment": value.split(";")[0]}

        coords = re.findall(r"[-+]?\d+\.\d+", value)

        for key, val in zip(
            ["x", "y", "z"],
            coords,
            strict=False,
        ):
            args[key] = val

        args["parent"] = None
        return args

    @model_serializer(mode="wrap")
    def serialize_to_str(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, object]:
        serialized = handler(self)
        for key, val in serialized.items():
            serialized[key] = str(val)

        return serialized

    @field_validator("*", mode="after")
    @classmethod
    def save_to_visual_parameters(cls, value, info: ValidationInfo):
        # Skip during initialization
        if len(cls.model_fields.keys()) != len(info.data) + 1:
            return value

        if info.field_name == "parent":
            CONTEXT_PARENT.set(value)
            return value

        if info.field_name is None:
            return value

        alias = cls.model_fields[info.field_name].alias  # pylint: disable=unsubscriptable-object

        if alias is None:
            return value

        element = CONTEXT_PARENT.get().xml.find(alias)

        if element is not None:
            element.text = str(value)
            CONTEXT_PARENT.get().parent.workspace.update_attribute(
                CONTEXT_PARENT.get(), "values"
            )

        return value


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
    def geometry(self):
        """
        Define the geometry of the Maxwell Plate.
        """
        if self._geometry is None:
            if self.visual_parameters is not None:
                xml = self.visual_parameters.xml
                self._geometry = PlateGeometry(
                    parent=self.visual_parameters,
                    **{child.tag: child.text for child in xml},
                )

        return self._geometry
