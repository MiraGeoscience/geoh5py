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
from typing import Any

import numpy as np
from PIL import Image

from .data import Data


class TextureData(Data):
    """
    Data container an image texture associated with vertices.

    :param texture_image: The texture image as bytes representation of a :obj:`PIL.Image` object.
    :param values: Record array mapping the pixel position to the vertices of the parent object.
    """

    _attribute_map = Data._attribute_map.copy()
    __VALUES_DTYPE = np.dtype([("v[0]", "<f4"), ("v[1]", "<f4")])

    def __init__(
        self,
        allow_move=False,
        texture_image: Image.Image | bytes | np.ndarray | None = None,
        values: np.recarray | None = None,
        **kwargs,
    ):
        self._texture_image: bytes | None = None
        super().__init__(allow_move=allow_move, values=values, **kwargs)
        self.texture_image = texture_image

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

        self._texture_image = value

        if self.on_file:
            self.workspace.update_attribute(self, "texture_image")

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
            if values.ndim != 2 or values.shape[1] != 2:
                raise ValueError("'values' requires an ndarray of shape (*, 2).")

            values = np.asarray(
                np.rec.fromarrays(
                    values.T.tolist(),
                    dtype=self.__VALUES_DTYPE,
                )
            )

        if values.dtype != self.__VALUES_DTYPE:
            raise TypeError(
                f"Array of 'values' must be of dtype = {self.__VALUES_DTYPE}"
            )

        if self.parent is not None and len(values) != self.parent.n_vertices:
            raise ValueError(
                "The length of 'values' must match the number of vertices in the parent entity."
            )

        return values
