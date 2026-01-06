# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2025-2026 Mira Geoscience Ltd.                                '
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
from pydantic import (
    AliasChoices,
    AliasGenerator,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)
from pydantic.alias_generators import to_pascal, to_snake

from .points import Points


if TYPE_CHECKING:
    from numpy import ndarray


class TextEntry(BaseModel):
    """
    Core parameters for text mesh data.

    :param starting_point: Starting point of the text.
    :param annotation: Annotation ID.
    :param color: Color of the text stored as a hex string.
    :param direction: Normal vector defining the direction of the text.
    :param font: Font type.
    :param font_size: Font size.
    :param layer_name: Layer name.
    :param normal: Normal vector of the text.
    :param obliquing_angle: Obliquing angle of the text.
    :param rotation_angle: Rotation angle of the text.
    :param text: Text string displayed.
    :param width_scale_factor: Width scale factor of the text.
    """

    model_config = ConfigDict(
        alias_generator=AliasGenerator(  # type: ignore
            validation_alias=to_snake,
            serialization_alias=to_pascal,
        ),
        validate_assignment=True,
    )

    starting_point: str
    annotation: int = 0
    color: str = "#ffffff"
    direction: str = "{1,0,0}"
    font: str = "builtin_simple"
    font_size: int = 10
    layer_name: str = ""
    normal: str = "{0,0,1}"
    obliquing_angle: int = 0
    rotation_angle: int = 0
    text: str = "<no-data>"
    width_scale_factor: float = 1.0

    @field_validator("starting_point", "normal", "direction", mode="before")
    @classmethod
    def array_to_str(cls, float_list: np.ndarray | list) -> str:
        """
        Convert a list of floats to a string representation.
        """
        if isinstance(float_list, tuple) and len(float_list) == 1:
            float_list = float_list[0]

        if isinstance(float_list, np.ndarray | list):
            float_list = "{" + ",".join([str(val) for val in float_list]) + "}"

        return float_list


class TextMesh(BaseModel):
    """
    Text mesh data.

    :param data: List of text entries with parameters.
    """

    data: list[TextEntry] = Field(
        validation_alias=AliasChoices("data", "Text Data"),
        serialization_alias="Text Data",
    )


class TextObject(Points):
    """
    Text object for annotated text labels in viewport.

    :param vertices: Vertices of the text object.
    :param text_mesh_data: Text mesh data as a JSON string or dictionary.
    """

    _attribute_map: dict = Points._attribute_map.copy()
    _attribute_map.update({"TextMesh Data": "text_mesh_data"})
    _default_name = "Text Object"
    _TYPE_UID: uuid.UUID = uuid.UUID("{c00905d1-bc3b-4d12-9f93-07fcf1450270}")

    def __init__(
        self,
        vertices: np.ndarray | list | tuple | None = None,
        text_mesh_data: str | None = None,
        **kwargs,
    ):
        self._text_mesh_data: TextMesh

        super().__init__(vertices=vertices, text_mesh_data=text_mesh_data, **kwargs)

    def copy(
        self,
        parent=None,
        *,
        copy_children: bool = True,
        clear_cache: bool = False,
        mask: np.ndarray | None = None,
        **kwargs,
    ) -> Points:
        """
        Sub-class extension of :func:`~geoh5py.objects.points.Points.copy`.
        """
        mask = self.validate_mask(mask)
        if mask is not None:
            text_data = TextMesh(
                data=[
                    elem
                    for elem, logic in zip(self.text_mesh_data.data, mask, strict=False)
                    if logic
                ]
            )
            kwargs.update({"text_mesh_data": text_data})

        new_entity = super().copy(
            parent=parent,
            copy_children=copy_children,
            clear_cache=clear_cache,
            mask=mask,
            **kwargs,
        )
        return new_entity

    @property
    def text_mesh_data(self) -> TextMesh:
        """
        Text mesh data.
        """

        return self._text_mesh_data

    @text_mesh_data.setter
    def text_mesh_data(self, value: str | dict | TextMesh | None):
        """
        Set the text mesh data.
        """
        if value is None:
            self._text_mesh_data = TextMesh(
                data=[
                    TextEntry(text="<no-data>", starting_point=vert)
                    for vert in self.vertices
                ]
            )
            return

        if isinstance(value, str):
            t_m_data = loads(value)
        else:
            t_m_data = value

        if isinstance(t_m_data, dict):
            t_m_data = TextMesh(**t_m_data)

        if not isinstance(t_m_data, TextMesh):
            raise TypeError("The 'Text Data' must be a dictionary or a JSON string.")

        if len(t_m_data.data) != self.n_vertices:
            raise ValueError(
                "The 'Text Data' dictionary must contain a list of len('n_vertices')."
            )

        self._text_mesh_data = t_m_data

    def _validate_value_length(self, values) -> list:
        """
        Validate the length of the values.

        :param values: Values to validate stored as a list or numpy array.
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

            for entry, elem in zip(self.text_mesh_data.data, value, strict=False):
                setattr(entry, key, elem)

            if self.on_file:
                self.workspace.update_attribute(self, "attributes")

        else:
            super().__setattr__(key, value)
