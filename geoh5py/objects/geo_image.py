#  Copyright (c) 2024 Mira Geoscience Ltd.
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

import uuid
import warnings
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image
from PIL.TiffImagePlugin import TiffImageFile

from ..data import FilenameData
from ..shared.conversion import GeoImageConversion
from ..shared.utils import (
    PILLOW_ARGUMENTS,
    box_intersect,
    dip_points,
    xy_rotation_matrix,
)
from .object_base import ObjectBase


if TYPE_CHECKING:
    from ..objects import Grid2D


class GeoImage(ObjectBase):  # pylint: disable=too-many-public-methods
    """
    Image object class.

    The GeoImage object position is defined by the four corner vertices.
    The values displayed in the image are stored in a separate entity, called
    'GeoImageMesh_Image', and stored as 'GeoImage.image_data' attribute. The image
    values themselves can be accessed through the 'GeoImage.image' attribute.

    The 'image' data can be set with:
        - A File on disk
        - An array of values defining the pixels of the image
            - A 2D array of values will create a grayscale image.
            - A 3D array of values will create an RGB image
        - A PIL.Image object

    Setting the 'image' property will create a 'GeoImageMesh_Image' entity and remove
    the previous one.

    :param cells: Array of indices defining segments connecting vertices.
    :param dip: Dip of the image in degrees from the vertices position.
    :param image: Image data as a numpy array, PIL.Image, bytes, or path to an image file.
    :param rotation: Rotation of the image in degrees, counter-clockwise.
    :param vertices: Array of vertices defining the four corners of the image.
    """

    __VERTICES_DTYPE = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8")])
    _TYPE_UID = uuid.UUID(
        fields=(0x77AC043C, 0xFE8D, 0x4D14, 0x81, 0x67, 0x75E300FB835A)
    )
    _converter: type[GeoImageConversion] = GeoImageConversion

    def __init__(
        self,
        *,
        cells: np.ndarray | list | tuple | None = None,
        dip: float | None = None,
        image: str | np.ndarray | BytesIO | Image.Image | FilenameData | None = None,
        rotation: float | None = None,
        vertices: np.ndarray | list | tuple | None = None,
        **kwargs,
    ):
        self._cells: np.ndarray | None
        self._vertices: np.ndarray | None
        self._tag: dict[int, Any] | None = None

        super().__init__(**kwargs)
        self._image_data: FilenameData | None = None
        self.vertices = vertices
        self.image = image
        self.cells = cells

        if rotation is not None:
            self.rotation = rotation

        if dip is not None:
            self.dip = dip

    @property
    def cells(self) -> np.ndarray:
        """
        Array of indices defining segments connecting vertices.
        """
        if self._cells is None and self.on_file:
            self._cells = self.workspace.fetch_array_attribute(self)

        return self._cells

    @cells.setter
    def cells(self, indices: np.ndarray | list | tuple | None):
        if isinstance(indices, (list, tuple)):
            indices = np.array(indices, ndmin=2)

        if indices is None:
            indices = np.c_[[0, 1, 2, 0], [0, 2, 3, 0]].T.astype("uint32")

        if not isinstance(indices, np.ndarray):
            raise TypeError(
                "Attribute 'cells' must be provided as type numpy.ndarray, list or tuple."
            )

        if indices.ndim != 2 or indices.shape != (2, 4):
            raise ValueError("Array of cells should be of shape (2, 4).")

        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("Indices array must be of integer type")

        self._cells = indices

        if self.on_file:
            self.workspace.update_attribute(self, "cells")

    def copy(
        self,
        parent=None,
        *,
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

    def copy_from_extent(
        self,
        extent: np.ndarray,
        parent=None,
        *,
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

        # transform the image to a grid
        grid = self.to_grid2d(parent=parent, mode="RGBA")

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
            grid.workspace.remove_entity(grid)
            warnings.warn("Image could not be cropped.")
            return None

        # transform the grid back to an image
        image_transformed = grid_transformed.to_geoimage(
            keys=grid_transformed.get_data_list(), mode="RGBA", normalize=False
        )

        grid.workspace.remove_entity(grid_transformed)
        grid.workspace.remove_entity(grid)

        return image_transformed

    @property
    def default_vertices(self) -> np.ndarray:
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
            ).astype(float)
        return np.asarray(
            [
                [0, 1, 0],
                [1, 1, 0],
                [1, 0, 0],
                [0, 0, 0],
            ]
        ).astype(float)

    @property
    def dip(self) -> float:
        """
        Calculated dip of the image in degrees from the vertices position.
        :return: the dip angle.
        """
        # Get rotation matrix
        rotation_matrix = xy_rotation_matrix(np.deg2rad(-self.rotation))

        # Rotate the vertices
        rotated_vertices = (rotation_matrix @ self.vertices.T).T

        # Calculate the vector perpendicular to the rotation
        delta_xyz = rotated_vertices[0] - rotated_vertices[3]

        # Compute dip in degrees
        dip = np.rad2deg(
            np.arctan2(delta_xyz[2], np.sqrt(delta_xyz[0] ** 2 + delta_xyz[1] ** 2))
        )

        return dip

    @dip.setter
    def dip(self, new_dip: float):
        # Transform the vertices to a plane
        self.vertices = (
            dip_points(
                self.vertices - self.origin,
                np.deg2rad(new_dip - self.dip),
                np.deg2rad(self.rotation),
            )
            + self.origin
        )

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
            param_x[0] + corners @ param_x[1:],
            param_y[0] + corners @ param_y[1:],
            param_z[0] + corners @ param_z[1:],
        ]

        self.set_tag_from_vertices()

    def georeferencing_from_image(self):
        """
        Georeferencing the GeoImage from the image.
        """
        if self.image is not None:
            if self.tag is not None:
                self.vertices = self.default_vertices
                self.georeferencing_from_tiff()
            else:
                self.vertices = self.default_vertices

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
            warnings.warn("The 'tif.' image has no referencing information.")

    @property
    def image(self):
        """
        Get the image as a :obj:`PIL.Image` object.
        """
        if self.image_data is not None and self.image_data.file_bytes is not None:
            return Image.open(BytesIO(self.image_data.file_bytes))

        return None

    @image.setter
    def image(
        self, image: str | np.ndarray | BytesIO | Image.Image | FilenameData | None
    ):
        if self._image_data is not None:
            raise AttributeError(
                "The 'image' property cannot be reset. "
                "Consider creating a new object."
            )

        if isinstance(image, (FilenameData, type(None))):
            self._image_data = image
            return

        image = self.validate_image_data(image)

        with TemporaryDirectory() as tempdir:
            if image.mode not in PILLOW_ARGUMENTS:
                raise NotImplementedError(
                    f"The mode {image.mode} of the image is not supported."
                )

            temp_file = Path(tempdir) / "image"
            image.save(temp_file, **PILLOW_ARGUMENTS[image.mode])
            image_file = self.add_file(str(temp_file))
            image_file.name = "GeoImageMesh_Image"
            image_file.entity_type.name = "GeoImageMesh_Image"

        self._image_data = image_file

        if self._vertices is None:
            self.vertices = self.default_vertices

        # if the image is a tiff save tag information
        if isinstance(image, TiffImageFile):
            self.tag = image
            self.to_grid2d(name=self.name + "_grid2d")

    @property
    def image_data(self) -> FilenameData | None:
        """
        Get the FilenameData entity holding the image.
        """
        if self._image_data is None:
            for child in self.children:
                if (
                    isinstance(child, FilenameData)
                    and child.name == "GeoImageMesh_Image"
                ):
                    self._image_data = child

        return self._image_data

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

    def mask_by_extent(
        self, extent: np.ndarray, inverse: bool = False
    ) -> np.ndarray | None:
        """
        Sub-class extension of :func:`~geoh5py.shared.entity.Entity.mask_by_extent`.

        Uses the four corners of the image to determine overlap with the extent window.
        """
        if self.extent is None or not box_intersect(self.extent, extent):
            return None

        return np.ones(self.vertices.shape[0], dtype=bool)

    @property
    def n_cells(self):
        """
        Number of vertices
        """
        return self.cells.shape[0]

    @property
    def n_vertices(self):
        """
        Number of vertices
        """
        return self.vertices.shape[0]

    @property
    def origin(self) -> np.array:
        """
        The origin of the image.
        :return: an array of the origin of the image in x, y, z.
        """

        return self.vertices[3, :]

    @property
    def rotation(self) -> float:
        """
        The rotation of the image in degrees, counter-clockwise.
        :return: the rotation angle.
        """
        dxy = np.r_[np.diff(self.vertices[:2, 0]), np.diff(self.vertices[:2, 1])]
        dxy /= np.linalg.norm(dxy)
        rotation_rad = np.arctan2(dxy[1], dxy[0])

        return np.rad2deg(rotation_rad)

    @rotation.setter
    def rotation(self, new_rotation):
        # Compute rotation matrix
        rotation_matrix = xy_rotation_matrix(np.deg2rad(new_rotation - self.rotation))

        # get the vertices without the origin
        vertices = self.vertices - self.origin

        # get the rotation matrix
        vertices = rotation_matrix @ vertices.T

        # save the vertices
        self.vertices = vertices.T + self.origin

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
        if path != "" and not Path(path).is_dir():
            raise FileNotFoundError(f"No such file or directory: {path}")

        if name.endswith((".tif", ".tiff")) and self.tag is not None:
            # save the image
            image: Image = self.image_georeferenced
            image.save(Path(path) / name, exif=image.getexif())
        else:
            self.image.save(Path(path) / name)

    def set_tag_from_vertices(self):
        """
        If tag is None, set the basic tag values based on vertices
        in order to export as a georeferenced .tiff.
        WARNING: this function must be used after georeference().
        """
        if self._tag is None:
            self._tag = {}

        if self.image is None:
            raise AttributeError("An 'image' must be set before georeferencing.")

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
    def tag(self) -> dict | None:
        """
        Georeferencing information of a tiff image stored in the header.
        :return: a dictionary containing the PIL.Image.tag information.
        """
        if self._tag:
            return self._tag.copy()
        return None

    @tag.setter
    def tag(self, value: Image.Image | dict | None):
        if isinstance(value, (Image.Image, TiffImageFile)):
            self._tag = dict(value.tag)
        elif isinstance(value, dict):
            self._tag = value
        elif value is None:
            self._tag = None
        else:
            raise ValueError("Input 'tag' must be a PIL.Image")

    def to_grid2d(
        self,
        mode: str | None = None,
        **grid2d_kwargs,
    ) -> Grid2D:
        """
        Create a geoh5py :obj:geoh5py.objects.grid2d.Grid2D from the geoimage in the same workspace.

        :param mode: The output image mode, defaults to the incoming image.mode.
            If "GRAY" convert the image to grayscale.
        :param grid2d_kwargs: Keyword arguments to pass to the
            :obj:`geoh5py.objects.grid2d.Grid2D` constructor.

        :return: the new created :obj:`geoh5py.objects.grid2d.Grid2D`.
        """
        return self.converter.to_grid2d(self, mode, **grid2d_kwargs)

    def validate_image_data(
        self, image: str | np.ndarray | BytesIO | Image.Image | FilenameData | None
    ) -> Image.Image:
        """
        Validate the input image.

        :param image: Image to validate.

        :return: Image converted to FileNameData object.
        """
        # todo: this should be changed in the future to accept tiff images
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
            if not Path(image).is_file():
                raise ValueError(f"Input image file {image} does not exist.")

            image = Image.open(image)
        elif isinstance(image, bytes):
            image = Image.open(BytesIO(image))

        elif not isinstance(image, Image.Image):
            raise ValueError(
                "Input 'value' for the 'image' property must be "
                "a 2D or 3D numpy.ndarray, bytes, PIL.Image or a path to an existing image."
                f"Get type {type(image)} instead."
            )

        return image

    @property
    def vertices(self) -> np.ndarray:
        """
        :obj:`~geoh5py.objects.object_base.ObjectBase.vertices`:
        Defines the four corners of the geo_image
        """
        # Case the vertices were removed from the object
        if self._vertices is None and self.on_file:
            self._vertices = self.workspace.fetch_array_attribute(self, "vertices")

        # Case where the vertices are not set but the image is defined
        if self._vertices is None and self.tag is not None and self.image is not None:
            self.vertices = self.default_vertices
            self.georeferencing_from_tiff()

        # Case neither vertices nor image are set
        if self._vertices is None:
            return self.default_vertices

        return self._vertices.view("<f8").reshape((-1, 3)).astype(float)

    @vertices.setter
    def vertices(self, xyz: np.ndarray | list | None):
        if xyz is None:
            self._vertices = None
            return

        if isinstance(xyz, list | tuple):
            xyz = np.array(xyz, ndmin=2)

        if not isinstance(xyz, np.ndarray):
            raise TypeError(
                "Input 'vertices' must be provided as type numpy.ndarray, list or tuple."
            )

        if np.issubdtype(xyz.dtype, np.number):
            xyz = np.asarray(
                np.core.records.fromarrays(xyz.T, dtype=self.__VERTICES_DTYPE)
            )

        if xyz.dtype != self.__VERTICES_DTYPE:
            raise TypeError(
                f"Array of 'vertices' must be of dtype = {self.__VERTICES_DTYPE}"
            )

        if len(xyz) != 4:
            raise ValueError("Array of 'vertices' must be of shape (4, 3).")

        self._vertices = xyz
        self._tag = None

        if self.on_file:
            self.workspace.update_attribute(self, "vertices")
