#  Copyright (c) 2023 Mira Geoscience Ltd.
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
import warnings
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from PIL import Image
from PIL.TiffImagePlugin import TiffImageFile

from .. import objects
from ..data import FilenameData
from ..shared.conversion import GeoImageConversion
from ..shared.utils import box_intersect
from .object_base import ObjectBase, ObjectType


class GeoImage(ObjectBase):
    """
    Image object class.

    .. warning:: Not yet implemented.

    """

    __TYPE_UID = uuid.UUID(
        fields=(0x77AC043C, 0xFE8D, 0x4D14, 0x81, 0x67, 0x75E300FB835A)
    )

    _converter = GeoImageConversion

    def __init__(self, object_type: ObjectType, **kwargs):
        self._vertices: None | np.ndarray = None
        self._cells = None
        self._tag: dict[int, Any] | None = None
        self._rotation: None | float = None

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

    def copy(
        self,
        parent=None,
        copy_children: bool = True,
        clear_cache: bool = False,
        mask: np.ndarray | None = None,
        **kwargs,
    ):
        """
        Function to copy an entity to a different parent entity.

        :param parent: New parent for the copied object.
        :param copy_children: Copy children entities.
        :param clear_cache: Clear cache of data values.
        :param mask: Array of indices to sub-sample the input entity.
        :param kwargs: Additional keyword arguments.
        """
        if mask is not None:
            warnings.warn("Masking is not supported for GeoImage objects.")

        new_entity = super().copy(
            parent=parent,
            copy_children=copy_children,
            clear_cache=clear_cache,
            **kwargs,
        )

        return new_entity

    def copy_from_extent(  # pylint: disable=too-many-locals
        self,
        extent: np.ndarray,
        parent=None,
        copy_children: bool = True,
        clear_cache: bool = False,
        inverse: bool = False,
        **kwargs,
    ) -> GeoImage | None:
        """
        Sub-class extension of :func:`~geoh5py.shared.entity.Entity.copy_from_extent`.
        """
        # todo: save the temp grid in a temp workspace?
        if self.vertices is None:
            raise AttributeError("Vertices are not defined.")

        if self.image is None:
            warnings.warn("Image is not defined.")
            return None

        # define the type of the image
        n_channel = len(self.image.mode)
        if n_channel == 1:
            transform = "GRAY"
        elif n_channel == 3:
            transform = "RGB"
        elif n_channel == 4:
            transform = "CMYK"
        else:
            warnings.warn(f"Image mode not supported (channels: {n_channel}).")
            return None

        # transform the image to a grid
        grid = self.to_grid2d(transform=transform)

        # transform the image
        grid_transformed = grid.copy_from_extent(
            extent=extent,
            parent=parent,
            copy_children=copy_children,
            clear_cache=clear_cache,
            inverse=inverse,
            from_image=True,
            **kwargs,
        )

        if grid_transformed is None:
            warnings.warn("Image could not be cropped.")
            return None

        # transform the grid back to an image
        image_transformed = grid_transformed.to_geoimage(
            keys=grid_transformed.get_data_list(), normalize=False
        )

        return image_transformed

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

    @property
    def extent(self) -> np.ndarray | None:
        """
        Geography bounding box of the object.

        :return: shape(2, 3) Bounding box defined by the bottom South-West and
            top North-East coordinates.
        """
        if self.vertices is not None:
            return np.c_[self.vertices.min(axis=0), self.vertices.max(axis=0)].T

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
            value = image
            if image.min() < 0 or image.max() > 255 or image.dtype != "uint8":
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

    def mask_by_extent(
        self, extent: np.ndarray, inverse: bool = False
    ) -> np.ndarray | None:
        """
        Sub-class extension of :func:`~geoh5py.shared.entity.Entity.mask_by_extent`.

        Uses the four corners of the image to determine overlap with the extent window.
        """
        if self.extent is None or not box_intersect(self.extent, extent):
            return None

        if self.vertices is not None:
            return np.ones(self.vertices.shape[0], dtype=bool)

        return None

    def set_tag_from_vertices(self):
        """
        If tag is None, set the basic tag values based on vertices
        in order to export as a georeferenced .tiff.
        WARNING: this function must be used after georeference().
        """

        if self.image is None:
            raise AttributeError("There is no image to reference")

        if not isinstance(self.vertices, np.ndarray):
            raise AttributeError("Vertices must be set before setting tag")

        if self._tag is None:
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
        Georeferencing information of a tiff image stored in the header.
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
        Get the geographic information from the PIL Image to georeference it.
        Run the georeference() method of the object.
        """
        if self.tag is None:
            raise AttributeError("The image is not georeferenced")

        try:
            # get geographic information
            u_origin = float(self.tag[33922][3])
            v_origin = float(self.tag[33922][4])
            u_cell_size = float(self.tag[33550][0])
            v_cell_size = float(self.tag[33550][1])
            u_count = float(self.tag[256][0])
            v_count = float(self.tag[257][0])
            u_oposite = float(u_origin + u_cell_size * u_count)
            v_oposite = float(v_origin - v_cell_size * v_count)

            # prepare georeferencing
            reference = np.array([[0.0, v_count], [u_count, v_count], [u_count, 0.0]])

            locations = np.array(
                [
                    [u_origin, v_origin, 0.0],
                    [u_oposite, v_origin, 0.0],
                    [u_oposite, v_oposite, 0.0],
                ]
            )

            # georeference the raster
            self.georeference(reference, locations)
        except KeyError:
            warnings.warn("The 'tif.' image has no referencing information")

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

    def save_as(self, name: str, path: str | Path = ""):
        """
        Function to save the geoimage into an image file.
        It the name ends by '.tif' or '.tiff' and the tag is not None
        then the image is saved as georeferenced tiff image ;
        else, the image is save with PIL.Image's save function.
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
        if not isinstance(path, (str, Path)):
            raise TypeError(
                f"The 'path' has to be a string or a Path; a '{type(name)}' was entered instead"
            )
        if path != "" and not os.path.isdir(path):
            raise FileNotFoundError(f"No such file or directory: {path}")

        if name.endswith((".tif", ".tiff")) and self.tag is not None:
            # save the image
            image: Image = self.image_georeferenced
            image.save(os.path.join(path, name), exif=image.getexif())
        else:
            self.image.save(os.path.join(path, name))

    def to_grid2d(
        self,
        transform: str = "GRAY",
        **grid2d_kwargs,
    ) -> objects.Grid2D:
        """
        Create a geoh5py :obj:geoh5py.objects.grid2d.Grid2D from the geoimage in the same workspace.
        :param transform: the type of transform ; if "GRAY" convert the image to grayscale ;
        if "RGB" every band is sent to a data of a grid.
        :return: the new created :obj:`geoh5py.objects.grid2d.Grid2D`.
        """
        return self.converter.to_grid2d(self, transform, **grid2d_kwargs)

    @property
    def rotation(self) -> float | None:
        """
        The rotation of the image in degrees
        :return: the rotation angle.
        """
        if self._rotation is None and self._vertices is not None:
            # Get the x and y values of the first and second corners
            corner_1_x = self._vertices["x"][0]
            corner_1_y = self._vertices["y"][0]
            corner_2_x = self._vertices["x"][1]
            corner_2_y = self._vertices["y"][1]

            # Form them into numpy arrays
            corner_1 = np.array([corner_1_x, corner_1_y])
            corner_2 = np.array([corner_2_x, corner_2_y])

            # Get the vector from corner_1 to corner_2
            delta = corner_2 - corner_1

            # Reference direction (the x-axis)
            reference = np.array([1, 0])

            # Calculate the angle between the reference direction and the vector
            rotation_rad = np.arctan2(
                np.cross(reference, delta), np.dot(reference, delta)
            )

            # Convert to degrees
            rotation_deg = np.rad2deg(rotation_rad)

            # Adjust the angle to lie between [0, 360)
            if rotation_deg < 0:
                rotation_deg = 360 + rotation_deg

            self._rotation = round(rotation_deg, 2)

        return self._rotation

    @rotation.setter
    def rotation(self, new_rotation):
        if new_rotation != 0 and self.vertices is not None:
            # Compute current rotation
            current_rotation = self.rotation
            if current_rotation is None:
                current_rotation = 0

            # Compute rotation angle in degrees
            rotation_angle_deg = new_rotation - current_rotation

            # Use the origin vertex (vertices[3]) as the center of rotation
            center = np.array([self.vertices["x"][3], self.vertices["y"][3]])

            # Move vertices so that center is at origin
            vertices_centered_x = self.vertices["x"] - center[0]
            vertices_centered_y = self.vertices["y"] - center[1]
            vertices_centered = np.array([vertices_centered_x, vertices_centered_y]).T

            # Compute rotation matrix using numpy
            rotation_rad = np.radians(rotation_angle_deg)
            rotation_matrix = np.array(
                [
                    [np.cos(rotation_rad), -np.sin(rotation_rad)],
                    [np.sin(rotation_rad), np.cos(rotation_rad)],
                ]
            )

            # Apply rotation
            rotated_vertices_centered = vertices_centered @ rotation_matrix

            # Move vertices back so that center is at original location
            self.vertices["x"] = rotated_vertices_centered[:, 0] + center[0]
            self.vertices["y"] = rotated_vertices_centered[:, 1] + center[1]

            # Update the stored rotation
            self._rotation = new_rotation
