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
from tempfile import TemporaryDirectory
from warnings import warn

import numpy as np
from PIL import Image
from PIL.TiffImagePlugin import TiffImageFile

from ..data import FilenameData
from .grid2d import Grid2D
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
        self._tag: dict | None = None

        super().__init__(object_type, **kwargs)

        object_type.workspace._register_object(self)

    @property
    def cells(self) -> np.ndarray | None:
        r"""
        :obj:`numpy.ndarray` of :obj:`int`, shape (\*, 2):
        Array of indices defining segments connecting vertices. Defined based on
        :obj:`~geoh5py.objects.curve.Curve.parts` if set by the user.
        """
        if getattr(self, "_cells", None) is None:

            if self.on_file:
                self._cells = self.workspace.fetch_array_attribute(self)
            else:
                self.cells = np.c_[[0, 1, 2, 0], [0, 2, 3, 0]].T.astype("uint32")

        return self._cells

    @cells.setter
    def cells(self, indices):
        assert indices.dtype == "uint32", "Indices array must be of type 'uint32'"
        self._cells = indices
        self.workspace.update_attribute(self, "cells")

    @property
    def default_vertices(self):
        """
        Assign the default vertices based on image pixel count
        """
        if self.image is not None:
            return np.asarray(
                [
                    [0, self.image.size[1], 0],
                    [self.image.size[0], self.image.size[1], 0],
                    [self.image.size[0], 0, 0],
                    [0, 0, 0],
                ]
            )
        return None

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
        if isinstance(image, np.ndarray) and image.ndim in [2, 3]:
            if image.ndim == 3 and image.shape[2] != 3:
                raise ValueError(
                    "Shape of the 'image' must be a 2D or "
                    "a 3D array with shape(*,*, 3) representing 'RGB' values."
                )
            value = image.astype(float)
            value -= value.min()
            value *= 255.0 / value.max()
            value = value.astype("uint8")
            image = Image.fromarray(value)
        elif isinstance(image, str):
            if not os.path.exists(image):
                raise ValueError(f"Input image file {image} does not exist.")

            image = Image.open(image)

            # if the image is a tiff save tag information
            if isinstance(image, TiffImageFile):
                self.tag = image

        elif isinstance(image, bytes):
            image = Image.open(BytesIO(image))
        elif isinstance(image, TiffImageFile):
            self.tag = image
        elif not isinstance(image, Image.Image):
            raise ValueError(
                "Input 'value' for the 'image' property must be "
                "a 2D or 3D numpy.ndarray, bytes, PIL.Image or a path to an existing image."
                f"Get type {type(image)} instead."
            )

        with TemporaryDirectory() as tempdir:
            ext = getattr(image, "format")
            temp_file = os.path.join(
                tempdir, f"image.{ext.lower() if ext is not None else 'tiff'}"
            )
            image.save(temp_file)

            if self.image_data is not None:
                self.workspace.remove_entity(self.image_data)

            image = self.add_file(temp_file)
            image.name = "GeoImageMesh_Image"
            image.entity_type.name = "GeoImageMesh_Image"

    def georeference(self, reference: np.ndarray | list, locations: np.ndarray | list):
        """
        Georeference the image vertices (corners) based on input reference and
        corresponding world coordinates.

        :param reference: Array of integers representing the reference used as reference points.
        :param locations: Array of floats for the corresponding world coordinates
            for each input pixel.

        :return vertices: Corners (vertices) in world coordinates.
        """
        reference = np.asarray(reference)
        locations = np.asarray(locations)
        if self.image is None:
            raise AttributeError("An 'image' must be set before georeferencing.")

        if reference.ndim != 2 or reference.shape[0] < 3 or reference.shape[1] != 2:
            raise ValueError(
                "Input reference points must be a 2D array of shape(*, 2) "
                "with at least 3 control points."
            )

        if (
            locations.ndim != 2
            or reference.shape[0] != locations.shape[0]
            or locations.shape[1] != 3
        ):
            raise ValueError(
                "Input 'locations' must be a 2D array of shape(*, 3) "
                "with the same number of rows as the control points."
            )
        constant = np.ones(reference.shape[0])
        param_x, _, _, _ = np.linalg.lstsq(
            np.c_[constant, reference], locations[:, 0], rcond=None
        )
        param_y, _, _, _ = np.linalg.lstsq(
            np.c_[constant, reference], locations[:, 1], rcond=None
        )
        param_z, _, _, _ = np.linalg.lstsq(
            np.c_[constant, reference], locations[:, 2], rcond=None
        )

        corners = self.default_vertices[:, :2]

        self.vertices = np.c_[
            param_x[0] + np.dot(corners, param_x[1:]),
            param_y[0] + np.dot(corners, param_y[1:]),
            param_z[0] + np.dot(corners, param_z[1:]),
        ]

        self.set_tag_from_vertices()

    def set_tag_from_vertices(self):
        """
        If tag is None, set the basic tag values based on vertices
        in order to export as a georeferenced .tiff.
        WARNING: this function must be used after georeference().
        """
        if self.tag is None:
            self._tag = {}
            width, height = self.image.size
            self._tag[256] = (width,)
            self._tag[257] = (height,)
            self._tag[33922] = (
                0.0,
                0.0,
                0.0,
                self.vertices[0, 0],
                self.vertices[0, 1],
                self.vertices[0, 2],
            )
            self._tag[33550] = (
                abs(self.vertices[1, 0] - self.vertices[0, 0]) / width,
                abs(self.vertices[0, 1] - self.vertices[2, 1]) / height,
                0.0,
            )

    @property
    def vertices(self) -> np.ndarray | None:
        """
        :obj:`~geoh5py.objects.object_base.ObjectBase.vertices`:
        Defines the four corners of the geo_image
        """
        if (getattr(self, "_vertices", None) is None) and self.on_file:
            self._vertices = self.workspace.fetch_array_attribute(self, "vertices")

        if self._vertices is None and self.image is not None:
            if self.tag is not None:
                self.vertices = self.default_vertices
                self.georeferencing_from_tiff()
            else:
                self.vertices = self.default_vertices

        if self._vertices is not None:
            return self._vertices.view("<f8").reshape((-1, 3)).astype(float)

        return self._vertices

    @vertices.setter
    def vertices(self, xyz: np.ndarray | list):
        if isinstance(xyz, list):
            xyz = np.asarray(xyz)

        if not isinstance(xyz, np.ndarray) or xyz.shape != (4, 3):
            raise ValueError("Input 'vertices' must be a numpy array of shape (4, 3)")

        xyz = np.asarray(
            np.core.records.fromarrays(xyz.T, names="x, y, z", formats="<f8, <f8, <f8")
        )
        self._vertices = xyz
        self.workspace.update_attribute(self, "vertices")

    @property
    def tag(self) -> dict | None:
        """
        Return the tag information of the input image (if .tif).
        :return: a dictionary containing the PIL.Image.tag information.
        """
        if self._tag:
            return self._tag.copy()
        return None

    @tag.setter
    def tag(self, image: Image.Image | dict | None):
        if isinstance(image, (Image.Image, TiffImageFile)):
            self._tag = dict(image.tag)
        elif isinstance(image, dict):
            self._tag = image
        elif image is None:
            self._tag = None
        else:
            raise ValueError("Input 'tag' must be a PIL.Image")

    def georeferencing_from_tiff(self):
        """
        Get the geogrpahic information from the PIL Image to georeference it.
        Run the georefence() method of the object.
        :param image: a .tif image open with PIL.Image.
        """
        try:
            if self._tag:
                # get geographic information
                u_origin = self._tag[33922][3]
                v_origin = self._tag[33922][4]
                u_cell_size = self._tag[33550][0]
                v_cell_size = self._tag[33550][1]
                u_count = self._tag[256][0]
                v_count = self._tag[257][0]
                u_oposite = u_origin + u_cell_size * u_count
                v_oposite = v_origin - v_cell_size * v_count

                # prepare georeferencing
                reference = np.array(
                    [[0.0, v_count], [u_count, v_count], [u_count, 0.0]]
                )

                locations = np.array(
                    [
                        [u_origin, v_origin, 0.0],
                        [u_oposite, v_origin, 0.0],
                        [u_oposite, v_oposite, 0.0],
                    ]
                )

                # georeference the raster
                self.georeference(reference, locations)

        except (AttributeError, KeyError):
            warn("The 'tif.' image has no referencing information.")

    def to_grid2d(
        self,
        transform: str = "GRAY",
        **grid2d_kwargs,
    ) -> Grid2D:
        """
        Create a geoh5py :obj:geoh5py.objects.grid2d.Grid2D from the geoimage in the same workspace.
        :param transform: the type of transform ; if "GRAY" convert the image to grayscale ;
        if "RGB" every band is sent to a data of a grid.
        :param grid2d_kwargs: **kwargs arguments of Grid2D: name, parent, rotation, dip, elevation.
        :return: the new created Grid2D.
        """
        if transform not in ["GRAY", "RGB"]:
            raise KeyError(
                f"'transform' has to be 'GRAY' or 'RGB', you entered {transform} instead."
            )
        if self.vertices is None:
            raise AttributeError("The 'vertices' has to be previously defined")

        # define name and elevation
        name = grid2d_kwargs.get("name", self.name)
        elevation = grid2d_kwargs.get("elevation", 0)

        # get geographic information
        u_origin = self.vertices[0, 0]
        v_origin = self.vertices[2, 1]
        u_count = self.default_vertices[1, 0]
        v_count = self.default_vertices[0, 1]
        u_cell_size = abs(u_origin - self.vertices[1, 0]) / u_count
        v_cell_size = abs(v_origin - self.vertices[0, 1]) / v_count

        # create the 2dgrid
        grid = Grid2D.create(
            self.workspace,
            origin=[u_origin, v_origin, elevation],
            u_cell_size=u_cell_size,
            v_cell_size=v_cell_size,
            u_count=u_count,
            v_count=v_count,
            **grid2d_kwargs,
        )

        # add the data to the 2dgrid
        value = Image.open(BytesIO(self.image_data.values))
        if transform == "GRAY":
            grid.add_data(
                data={
                    f"{name}_GRAY": {
                        "values": np.array(value.convert("L")).astype(np.uint32)[::-1],
                        "association": "CELL",
                    }
                }
            )
        elif transform == "RGB":
            if np.array(value).shape[-1] != 3:
                raise IndexError("To export to RGB the image has to have 3 bands")

            grid.add_data(
                data={
                    f"{name}_R": {
                        "values": np.array(value).astype(np.uint32)[::-1, :, 0],
                        "association": "CELL",
                    },
                    f"{name}_G": {
                        "values": np.array(value).astype(np.uint32)[::-1, :, 1],
                        "association": "CELL",
                    },
                    f"{name}_B": {
                        "values": np.array(value).astype(np.uint32)[::-1, :, 2],
                        "association": "CELL",
                    },
                }
            )
        return grid

    @property
    def image_georeferenced(self) -> Image.Image | None:
        """
        Get the image as a georeferenced :obj:`PIL.Image` object.
        """
        if self.tag is not None and self.image is not None:
            image = self.image

            # modify the exif
            for id_ in self.tag:
                image.getexif()[id_] = self.tag[id_]

            return image

        return None

    def save_as(self, name: str, path: str = ""):
        """
        Function to save the geoimage into a file.
        It the name ends by '.tif' or '.tiff' and the tag is not None
        :param name: the name to give to the image.
        :param path: the path of the file of the image, default: ''.
        """

        # verifications
        if self.image is None:
            raise AttributeError("The object contains no image data")
        if not isinstance(name, str):
            raise TypeError(
                f"The 'name' has to be a string; a '{type(name)}' was entered instead"
            )
        if not isinstance(path, str):
            raise TypeError(
                f"The 'path' has to be a string; a '{type(name)}' was entered instead"
            )
        if path != "" and not os.path.isdir(path):
            raise FileNotFoundError(f"No such file or directory: {path}")

        if name.endswith((".tif", ".tiff")) and self.tag is not None:
            # save the image
            image: Image = self.image_georeferenced
            image.save(os.path.join(path, name), exif=image.getexif())
        else:
            self.image.save(os.path.join(path, name))
