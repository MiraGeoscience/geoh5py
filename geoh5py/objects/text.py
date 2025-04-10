# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2025 Mira Geoscience Ltd.                                     '
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

import uuid
from json import loads
from typing import TYPE_CHECKING

import numpy as np
from pydantic import AliasChoices, BaseModel, Field, field_validator

from .points import Points


if TYPE_CHECKING:
    from numpy import ndarray


class TextEntry(BaseModel):
    """
    Core parameters for text mesh data.
    """

    starting_point: str = Field(
        validation_alias=AliasChoices("starting_point", "StartingPoint"),
        serialization_alias="StartingPoint",
    )
    annotation: int = Field(
        validation_alias=AliasChoices("annotation", "Annotation"),
        serialization_alias="Annotation",
        default=0,
    )
    color: str | None = Field(
        validation_alias=AliasChoices("color", "Color"),
        serialization_alias="Color",
        default="#ffffff",
    )
    direction: str | None = Field(
        validation_alias=AliasChoices("direction", "Direction"),
        serialization_alias="Direction",
        default="{1,0,0}",
    )
    font: str | None = Field(
        validation_alias=AliasChoices("font", "Font"),
        serialization_alias="Font",
        default="builtin_simple",
    )
    font_size: int | None = Field(
        validation_alias=AliasChoices("font_size", "FontSize"),
        serialization_alias="FontSize",
        default=10,
    )
    layer_name: str | None = Field(
        validation_alias=AliasChoices("layer_name", "LayerName"),
        serialization_alias="LayerName",
        default="",
    )
    normal: str | None = Field(
        validation_alias=AliasChoices("normal", "Normal"),
        serialization_alias="Normal",
        default="{0,0,1}",
    )
    obliquing_angle: int | None = Field(
        validation_alias=AliasChoices("obliquing_angle", "ObliquingAngle"),
        serialization_alias="ObliquingAngle",
        default=0,
    )
    rotation_angle: int | None = Field(
        validation_alias=AliasChoices("rotation_angle", "RotationAngle"),
        serialization_alias="RotationAngle",
        default=0,
    )
    text: str | None = Field(
        validation_alias=AliasChoices("text", "Text"),
        serialization_alias="Text",
        default="<no-data>",
    )
    width_scale_factor: float | None = Field(
        validation_alias=AliasChoices("width_scale_factor", "WidthScaleFactor"),
        serialization_alias="WidthScaleFactor",
        default=1.0,
    )

    @field_validator("starting_point", "normal", "direction", mode="before")
    @classmethod
    def array_to_str(cls, float_list: np.ndarray | list):
        if isinstance(float_list, np.ndarray | list):
            float_list = "{" + ",".join([str(val) for val in float_list]) + "}"

        return float_list


class TextData(BaseModel):
    """
    Text mesh data.
    """

    text_data: list[TextEntry] = Field(
        validation_alias=AliasChoices("text_data", "Text Data"),
        serialization_alias="Text Data",
    )


class TextObject(Points):
    """
    Text object for annotated text labels in viewport.
    """

    _attribute_map: dict = Points._attribute_map.copy()
    _attribute_map.update({"TextMesh Data": "text_mesh_data"})
    _default_name = "Text Object"
    _TYPE_UID: uuid.UUID | None = uuid.UUID("{c00905d1-bc3b-4d12-9f93-07fcf1450270}")

    def __init__(
        self,
        vertices: np.ndarray | list | tuple | None = None,
        text_mesh_data: str | None = None,
        **kwargs,
    ):
        self._text_mesh_data: TextData

        super().__init__(vertices=vertices, text_mesh_data=text_mesh_data, **kwargs)

    @property
    def text_mesh_data(self) -> TextData:
        """
        Text mesh data.
        """

        return self._text_mesh_data

    @text_mesh_data.setter
    def text_mesh_data(self, value: str | None):
        """
        Set the text mesh data.
        """

        self._text_mesh_data = self._validate_text_data(value)

    @property
    def extent(self):
        """
        Geography bounding box of the object.
        """
        return None

    def mask_by_extent(self, extent: ndarray, inverse: bool = False) -> None:
        """
        Sub-class extension of :func:`~geoh5py.shared.entity.Entity.mask_by_extent`.
        """
        return None

    def _validate_text_data(self, value: str | dict | None) -> TextData:
        """
        Validate the text data.
        """
        if value is None:
            return TextData(
                text_data=[
                    TextEntry(text="<no-data>", starting_point=vert)
                    for vert in self.vertices
                ]
            )

        if isinstance(value, str):
            value = loads(value)

        if not isinstance(value, dict):
            raise TypeError("The 'Text Data' must be a dictionary or a JSON string.")

        mesh_data = TextData(**value)

        if len(mesh_data.text_data) != self.n_vertices:
            raise ValueError(
                "The 'Text Data' dictionary must contain a list of len('n_vertices')."
            )

        return mesh_data

    def _validate_value_length(self, values) -> list:
        """
        Validate the length of the values.
        """
        if not isinstance(values, (list, np.ndarray)):
            values = [values] * self.n_vertices

        if len(values) != self.n_vertices:
            raise ValueError(
                "The 'Text Data' entries must contain a list of len('n_vertices')."
            )
        return values

    def __setattr__(self, key, value):
        """
        Set the attribute value.
        """
        if key in TextEntry.model_fields:
            value = self._validate_value_length(value)

            for entry, elem in zip(self.text_mesh_data.text_data, value, strict=False):
                entry.__setattr__(key, elem)

            if self.on_file:
                self.workspace.update_attribute(self, "attributes")

        else:
            super().__setattr__(key, value)
