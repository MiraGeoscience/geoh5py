#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoh5py.
#
#  geoh5py is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  geoh5py is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with geoh5py.  If not, see <https://www.gnu.org/licenses/>.
from __future__ import annotations

import os
import uuid
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from ..data import Data, FilenameData
from .object_base import ObjectBase, ObjectType


class GeoImage(ObjectBase):
    """
    Image object class.

    .. warning:: Not yet implemented.

    """

    __TYPE_UID = uuid.UUID(
        fields=(0x77AC043C, 0xFE8D, 0x4D14, 0x81, 0x67, 0x75E300FB835A)
    )

    def __init__(self, object_type: ObjectType, **kwargs):
        self._vertices = None
        self._cells = None

        super().__init__(object_type, **kwargs)

        self.entity_type.name = "GeoImage"

        object_type.workspace._register_object(self)

    @property
    def cells(self) -> np.ndarray | None:
        r"""
        :obj:`numpy.ndarray` of :obj:`int`, shape (\*, 2):
        Array of indices defining segments connecting vertices. Defined based on
        :obj:`~geoh5py.objects.curve.Curve.parts` if set by the user.
        """
        if getattr(self, "_cells", None) is None:

            if self.existing_h5_entity:
                self._cells = self.workspace.fetch_cells(self.uid)
            else:
                self._cells = np.c_[[0, 1, 2, 0], [0, 2, 3, 0]].T.astype("uint32")

        return self._cells

    @cells.setter
    def cells(self, indices):
        assert indices.dtype == "uint32", "Indices array must be of type 'uint32'"
        self.modified_attributes = "cells"
        self._cells = indices

    @property
    def vertices(self) -> np.ndarray | None:
        """
        :obj:`~geoh5py.objects.object_base.ObjectBase.vertices`:
        Defines the four corners of the geo_image
        """
        if (getattr(self, "_vertices", None) is None) and self.existing_h5_entity:
            self._vertices = self.workspace.fetch_coordinates(self.uid, "vertices")

        if self._vertices is not None:
            return self._vertices.view("<f8").reshape((-1, 3)).astype(float)

        return self._vertices

    @vertices.setter
    def vertices(self, xyz: np.ndarray):
        if not isinstance(xyz, np.ndarray) or xyz.shape != (4, 3):
            raise ValueError("Input 'vertices' must be a numpy array of shape (4, 3)")

        xyz = np.asarray(
            np.core.records.fromarrays(xyz.T, names="x, y, z", formats="<f8, <f8, <f8")
        )

        self.modified_attributes = "vertices"
        self._vertices = xyz

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID

    @property
    def image_data(self):
        """
        Get the FilenameData entity holding the image.
        """
        for child in self.children:
            if isinstance(child, FilenameData) and child.name == "GeoImageMesh_Image":
                return child
        return None

    @property
    def image(self):
        """
        Get the image as a :obj:`PIL.Image` object.
        """
        if self.image_data is not None:
            return Image.open(BytesIO(self.image_data.values))

        return None

    @image.setter
    def image(self, image: str | np.ndarray | BytesIO | Image.Image):
        """
        Create a :obj:`~geoh5py.data.filename_data.FilenameData`
        from dictionary of name and arguments.
        The provided arguments can be any property of the target Data class.

        :return: List of new Data objects.
        """
        if isinstance(image, np.ndarray):
            value = image.astype(float)
            value -= value.min()
            value *= 255.0 / value.max()
            value = value.astype("uint8")
            image = Image.fromarray(value)
        elif isinstance(image, str):
            if not os.path.exists(image):
                raise ValueError(f"Input image file {image} does not exist.")
            image = Image.open(image)
        elif isinstance(image, bytes):
            image = Image.open(BytesIO(image))
        elif not isinstance(image, Image.Image):
            raise UserWarning(
                "Input 'value' for the 'image' property must be "
                "a numpy.ndarray of values, bytes, PIL.Image or path to existing image."
            )

        with TemporaryDirectory() as tempdir:
            ext = getattr(image, "format")
            tempfile = (
                Path(tempdir) / f"image.{ext.lower() if ext is not None else 'tiff'}"
            )
            image.save(tempfile)

            with open(tempfile, "rb") as raw_binary:
                values = raw_binary.read()

        if self.image is None:
            attributes = {
                "name": "GeoImageMesh_Image",
                "file_name": self.name,
                "association": "OBJECT",
                "parent": self,
                "values": values,
            }
            entity_type = {"name": "GeoImageMesh_Image", "primitive_type": "FILENAME"}

            self.workspace.create_entity(
                Data, entity=attributes, entity_type=entity_type
            )
        else:
            self.image.values = values

    def georeference(self, pixels: np.ndarray | list, locations: np.ndarray | list):
        """
        Georeference the image vertices (corners) based on input pixels and
        corresponding world coordinates.

        :param pixels: Array of integers representing the pixels used as reference points.
        :param locations: Array of floats for the corresponding world coordinates
            for each input pixel.

        :return vertices: Corners (vertices) in world coordinates.
        """

        pixels = np.asarray(pixels)
        locations = np.asarray(locations)
        if self.shape is None:
            raise AttributeError("An 'image' must be set before georeferencing.")

        if pixels.shape[0] < 3:
            raise ValueError(
                "At least 3 reference points are needed for georeferencing of an image."
            )

        if pixels.shape[0] != locations.shape[0]:
            raise ValueError(
                f"Mismatch between the number of reference 'pixels' with shape {pixels.shape}"
                f" and number of 'locations' with shape {locations.shape}."
            )
        param_x, _, _, _ = np.linalg.lstsq(
            np.c_[np.ones(3), pixels], locations[:, 0], rcond=None
        )
        param_y, _, _, _ = np.linalg.lstsq(
            np.c_[np.ones(3), pixels], locations[:, 1], rcond=None
        )
        param_z, _, _, _ = np.linalg.lstsq(
            np.c_[np.ones(3), pixels], locations[:, 2], rcond=None
        )

        corners = np.vstack(
            [
                [0, self.shape[0]],
                [self.shape[1], self.shape[0]],
                [self.shape[1], 0],
                [0, 0],
            ]
        )

        self.vertices = np.c_[
            param_x[0] + np.dot(corners, param_x[1:]),
            param_y[0] + np.dot(corners, param_y[1:]),
            param_z[0] + np.dot(corners, param_z[1:]),
        ]

    @property
    def shape(self):
        """
        Image size
        """
        if self.image is not None:
            return self.image.size
        return None
