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

from io import BytesIO
from typing import Any, Self

import numpy as np
from PIL import Image
from pydantic import (
    AliasChoices,
    AliasGenerator,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from pydantic.alias_generators import to_pascal

from .data import Data


class CompressedTextures(BaseModel):
    """
    Data container for an image texture associated with vertices.

    :param textures: The texture images as bytes representation of a :obj:`PIL.Image` object.
    :param widths: Array of images widths with padding for compression.
    :param heights: Array of images heights with padding for compression.
    :param valid_widths: Array of images widths without padding.
    :param valid_heights: Array of images heights without padding.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        serialize_by_alias=True,
        alias_generator=AliasGenerator(
            serialization_alias=to_pascal,
        ),
    )

    formats: np.ndarray = Field(validation_alias=AliasChoices("Formats", "formats"))
    heights: np.ndarray = Field(validation_alias=AliasChoices("Heights", "heights"))
    textures: dict[str, bytes] = Field(
        validation_alias=AliasChoices("Textures", "textures")
    )
    valid_widths: np.ndarray = Field(
        validation_alias=AliasChoices("ValidWidths", "valid_widths")
    )
    valid_heights: np.ndarray = Field(
        validation_alias=AliasChoices("ValidHeights", "valid_heights")
    )
    widths: np.ndarray = Field(validation_alias=AliasChoices("Widths", "widths"))

    @field_validator("formats")
    @classmethod
    def validate_compression(cls, formats: np.ndarray) -> np.ndarray:
        if not np.all(np.isin(formats, [32849, 32856, 33776, 33779])):
            raise ValueError("Formats must be one of 32849, 32856, 33776, 33779")
        return formats

    @field_validator("textures", mode="before")
    @classmethod
    def validate_texture_images(
        cls, formats: dict[str, np.ndarray | bytes | Image.Image]
    ) -> dict[str, bytes]:
        for key, value in formats.items():
            formats[key] = TextureData.array_image_to_bytes(value)
        return formats

    @model_validator(mode="after")
    def arrays_size(self) -> Self:
        """
        Validate that all arrays have the same length as the number of textures.
        """
        for array in [
            self.formats,
            self.heights,
            self.valid_widths,
            self.valid_heights,
            self.widths,
        ]:
            if len(array) != len(self.textures):
                raise ValueError(
                    "All arrays must have the same length as the number of textures"
                )
        return self


class TextureData(Data):
    """
    Data container an image texture associated with vertices.

    :param texture_image: The texture image as bytes representation of a :obj:`PIL.Image` object.
    :param values: Record array mapping the pixel position to the vertices of the parent object.
    """

    _attribute_map = Data._attribute_map.copy()
    __VALUES_DTYPE = np.dtype([("v[0]", "<f4"), ("v[1]", "<f4")])
    __COMPRESSED_DTYPE = np.dtype([("v[0]", "<f4"), ("v[1]", "<f4"), ("v[2]", "<f4")])

    def __init__(
        self,
        allow_move=False,
        texture_image: Image.Image | bytes | np.ndarray | None = None,
        compressed_textures: CompressedTextures | None = None,
        values: np.recarray | None = None,
        **kwargs,
    ):

        self._texture_image: bytes | None = None
        self._compressed_textures: CompressedTextures | None = None

        super().__init__(allow_move=allow_move, values=values, **kwargs)

        self.texture_image = texture_image
        self.compressed_textures = compressed_textures

    @property
    def image(self) -> Image.Image | None:
        """
        Get the image as a :obj:`PIL.Image` object.
        """
        if self.texture_image is not None:
            return Image.open(BytesIO(self.texture_image))
        return None

    def _set_parent(self, parent):
        """
        Set the parent of the texture data.
        :param parent:
        :return:
        """
        if parent is not None and not hasattr(parent, "vertices"):
            raise TypeError("The parent of `texture_data` must have vertices.")

        super()._set_parent(parent)

    @property
    def compressed_textures(self) -> CompressedTextures | None:
        """
        The compressed textures associated with the vertices.
        """
        if self._compressed_textures is None and self.on_file:
            textures = self.workspace.fetch_compressed_textures(self.uid)

            if textures is not None:
                self._compressed_textures = CompressedTextures(**textures)

        return self._compressed_textures

    @compressed_textures.setter
    def compressed_textures(self, value: dict | CompressedTextures | None):
        if isinstance(value, dict):
            value = CompressedTextures(**value)

        if not isinstance(value, None | CompressedTextures):
            raise TypeError(
                "Attribute 'compressed_textures' must be a dict, TextureData or None."
            )

        self._compressed_textures = value

        if self.on_file:
            self.workspace.update_attribute(self, "compressed_textures")

    @property
    def texture_image(self) -> bytes | None:
        """
        The texture image associated with the vertices.
        """
        if self._texture_image is None and self.on_file:
            self._texture_image = self.workspace.fetch_file_object(
                self.uid, "TextureImage"
            )

        return self._texture_image

    @texture_image.setter
    def texture_image(self, value: np.ndarray | bytes | Image.Image | None):

        value = self.array_image_to_bytes(value)
        self._texture_image = value

        if self.on_file:
            self.workspace.update_attribute(self, "texture_image")

    @staticmethod
    def array_image_to_bytes(
        value: np.ndarray | bytes | Image.Image | None,
    ) -> bytes | None:
        if isinstance(value, np.ndarray):
            if value.ndim not in (2, 3) or (value.ndim == 3 and value.shape[2] != 3):
                raise ValueError(
                    "Shape of the 'texture_image' must be a 2D or "
                    "a 3D array with shape(*,*, 3) representing 'RGB' values."
                )

            if value.min() < 0 or value.max() > 255 or value.dtype != "uint8":
                value = value.astype(np.float64)
                value -= value.min()
                value *= 255.0 / value.max()
                value = value.astype("uint8")

            value = Image.fromarray(value)

        if isinstance(value, Image.Image):
            bio = BytesIO()
            value.save(bio, format="PNG")
            value = bio.getvalue()

        if not isinstance(value, (bytes, type(None))):
            raise TypeError(
                "Attribute 'texture_image' must be a numpy array, PIL.Image, bytes or None."
            )

        return value

    def validate_values(self, values: Any | None) -> np.ndarray | None:
        """
        Validate the shape and type of values describing the vector association
        between vertices and texture.

        :param values: Values to validate.
        """
        if values is None:
            return values

        if isinstance(values, (list, tuple)):
            values = np.array(values, ndmin=2)

        if not isinstance(values, np.ndarray):
            raise TypeError(
                "Attribute 'values' must be a list, tuple or numpy array. "
                f"Object of type {type(values)} provided."
            )

        if np.issubdtype(values.dtype, np.number):
            if values.ndim != 2 or values.shape[1] not in [2, 3]:
                raise ValueError(
                    "'values' requires an ndarray of shape (*, 2) or (*, 3)."
                )

            values = np.asarray(
                np.rec.fromarrays(
                    values.T.tolist(),
                    dtype=self.__VALUES_DTYPE
                    if values.shape[1] == 2
                    else self.__COMPRESSED_DTYPE,
                )
            )

        if values.dtype not in (self.__VALUES_DTYPE, self.__COMPRESSED_DTYPE):
            raise TypeError(
                "Array of 'values' must be of dtype "
                f"{self.__VALUES_DTYPE} or {self.__COMPRESSED_DTYPE}."
            )

        if self.parent is not None and len(values) != self.parent.n_vertices:
            raise ValueError(
                "The length of 'values' must match the number of vertices in the parent entity."
            )

        return values
