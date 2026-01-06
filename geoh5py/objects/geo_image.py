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

# pylint: disable=too-many-lines

from __future__ import annotations

import uuid
import warnings
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from ..data import FilenameData
from ..shared.conversion import GeoImageConversion
from ..shared.cut_by_extent import Plane
from ..shared.utils import (
    PILLOW_ARGUMENTS,
    are_affine,
    are_coplanar,
    box_intersect,
    xy_rotation_matrix,
    yz_rotation_matrix,
)
from .object_base import Entity, ObjectBase


if TYPE_CHECKING:
    from ..objects import Grid2D
    from ..workspace import Workspace


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

        if vertices is None:
            self.georeferencing_from_image()

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
        if self.image is None:
            warnings.warn("Image is not defined.")
            return None

        if inverse:
            raise NotImplementedError(
                "Inverse mask is not implemented yet with images."
            )

        # todo: image can contains several images attached as children
        if copy_children is False:
            warnings.warn(
                "The 'copy_children' argument is not applicable to GeoImage objects."
            )

        # 1. find the point where the image and extent intersect
        plane = Plane.from_points(self.vertices[3], self.vertices[2], self.vertices[0])

        new_extent, new_vertices = plane.extent_from_vertices_and_box(
            self,
            self.vertices,
            extent,
        )

        if new_extent is None or new_vertices is None:
            return None

        if np.isclose(new_vertices, self.vertices).all():
            return self.copy(parent, clear_cache=clear_cache)

        # 3. crop the image to the new extent (PIL image coordinates)
        cropped_image = self.image.crop(
            (
                new_extent[0],
                self.v_count - new_extent[3],
                new_extent[2],
                self.v_count - new_extent[1],
            )
        )

        # 4. get the final results
        kwargs.update(
            {
                "image": cropped_image,
                "vertices": new_vertices,
            }
        )

        return self._create_geoimage_from_attributes(parent, **kwargs)

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
            ).astype(np.float64)
        return np.asarray(
            [
                [0, 1, 0],
                [1, 1, 0],
                [1, 0, 0],
                [0, 0, 0],
            ]
        ).astype(np.float64)

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
        dip = np.arctan2(delta_xyz[2], delta_xyz[1])
        dip = np.round(np.rad2deg(dip), decimals=3)

        return dip % 360.0

    @dip.setter
    def dip(self, new_dip: float):
        vertices = self.vertices - self.origin
        vertices = xy_rotation_matrix(np.deg2rad(-self.rotation)) @ vertices.T
        # when we are defining 30 degree, we it do dip donward 30 degrees
        vertices = yz_rotation_matrix(np.deg2rad(new_dip - self.dip)) @ vertices
        vertices = xy_rotation_matrix(np.deg2rad(self.rotation)) @ vertices
        self.vertices = vertices.T + self.origin

    def georeference(
        self,
        tie_points: np.ndarray | list | tuple,
        u_cell_size: float | None = None,
        v_cell_size: float | None = None,
    ):
        """
        Georeference the image vertices (corners) based on input reference and
        corresponding world coordinates.

        :param tie_points: Array of tie points of shape (n, 2, 3)
            where n is the number of tie points, 2 corresponds to
            pixel and world coordinates, and 3 corresponds to (i, j, k)
            and (x, y, z) respectively,
            or a flat list/tuple of a 6*n length.

        :param u_cell_size: Cell size in the u direction. If None, it is
            computed from the current vertices and image size.
        :param v_cell_size: Cell size in the v direction. If None, it is
            computed from the current vertices and image size.

        :return vertices: Corners (vertices) in world coordinates.
        """

        if self.image is None:
            raise AttributeError("An 'image' must be set before georeferencing.")

        tie_points_verified = self._parse_tie_points(tie_points)

        corners = self._compute_image_corners(
            tie_points_verified, u_cell_size, v_cell_size
        )

        # Check if errors were returned
        if isinstance(corners, list):
            raise ValueError(f"Failed to compute image corners: {'; '.join(corners)}")

        self.vertices = corners

        self.set_tag_from_vertices()

    def georeferencing_from_image(self):
        """
        Georeferencing the GeoImage from the image.
        """
        if self.image is not None:
            self.vertices = self.default_vertices
            if self.tag is not None:
                self.georeferencing_from_tiff()

    def georeferencing_from_tiff(self):
        """
        Get the geographic information from the PIL Image to georeference it.
        Run the georeference() method of the object.
        """
        if self.tag is None or self.image is None:
            raise AttributeError("The image is not georeferenced")

        if not all(key in self.tag for key in (33550, 33922)):
            warnings.warn(
                "The 'tif.' image is missing one or more required tags "
                "(33550, 33922) for georeferencing."
            )
            return

        # get the tag values
        tie_points = self._parse_tie_points(self.tag[33922])
        u_cell_size = self.tag[33550][0]
        v_cell_size = self.tag[33550][1]

        vertices = self._compute_image_corners(tie_points, u_cell_size, v_cell_size)

        if isinstance(vertices, list):
            warning_message = (
                "Georeferencing from tiff failed because of the following reasons:"
                "\n - ".join(vertices)
            )
            warnings.warn(warning_message)
            return

        self.vertices = vertices

    @property
    def image(self):
        """
        Get the image as a :obj:`PIL.Image` object.
        """
        if self.image_data is not None and self.image_data.file_bytes is not None:
            old_limit = Image.MAX_IMAGE_PIXELS
            Image.MAX_IMAGE_PIXELS = None
            try:
                im = Image.open(BytesIO(self.image_data.file_bytes))
            finally:
                Image.MAX_IMAGE_PIXELS = old_limit
            return im

        return None

    @image.setter
    def image(
        self, image: str | np.ndarray | BytesIO | Image.Image | FilenameData | None
    ):
        if self._image_data is not None:
            raise AttributeError(
                "The 'image' property cannot be reset. Consider creating a new object."
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
        if hasattr(image, "tag") and image.tag:
            self.tag = image

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
    def origin(self) -> np.ndarray:
        """
        The origin of the image.
        :return: an array of the origin of the image in x, y, z.
        """
        return self.vertices[3, :]

    @property
    def rotation(self) -> float:
        """
        The rotation of the image in degrees, counter-clockwise.

        :raises: If the vertices do not form a rectangle with 90-degree angles,
        a ValueError is raised.

        :raises: If the image requires more than dip and rotation transformations,
        a ValueError is raised.

        :return: the rotation angle.
        """
        plane = Plane.from_points(self.vertices[3], self.vertices[2], self.vertices[0])
        if not plane.dip_rotation_only:
            raise ValueError(
                "The vertices do not define a rectangle that can be "
                "explained by rotation and dip only."
            )

        axes = self.vertices[2, :2] - self.origin[:2]
        return float(np.rad2deg(np.arctan2(axes[1], axes[0])))

    @rotation.setter
    def rotation(self, new_rotation):
        rotation_matrix = xy_rotation_matrix(np.deg2rad(new_rotation - self.rotation))
        vertices = self.vertices - self.origin
        vertices = (rotation_matrix @ vertices.T).T + self.origin
        self.vertices = vertices

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

        self._tag[256] = (self.u_count,)
        self._tag[257] = (self.v_count,)

        self._tag[33922] = tuple(
            v
            for (i, j, k), (x, y, z) in zip(
                self.default_vertices, self.vertices, strict=False
            )
            for v in (i, j, k, x, y, z)
        )

        self._tag[33550] = (
            self.u_cell_size,
            self.v_cell_size,
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
        if hasattr(value, "tag") and value.tag:  # type: ignore
            self._tag = dict(value.tag)  # type: ignore
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

    @staticmethod
    def validate_image_data(
        image: str | np.ndarray | BytesIO | Image.Image | FilenameData | None,
    ) -> Image.Image:
        """
        Validate the input image.

        :param image: Image to validate.

        :return: Image converted to FileNameData object.
        """
        # todo: this should be changed in the future to accept n dims tiff images
        if isinstance(image, np.ndarray) and image.ndim in [2, 3]:
            if image.ndim == 3 and image.shape[2] != 3:
                raise ValueError(
                    "Shape of the 'image' must be a 2D or "
                    "a 3D array with shape(*,*, 3) representing 'RGB' values."
                )
            value = image
            if image.min() < 0 or image.max() > 255 or image.dtype != "uint8":
                value = image.astype(np.float64)
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
            self.georeferencing_from_tiff()  # unlikely

        # Case neither vertices nor image are set
        if self._vertices is None:
            return self.default_vertices

        return self._vertices.view("<f8").reshape((-1, 3)).astype(np.float64)

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
            xyz = np.asarray(np.rec.fromarrays(xyz.T, dtype=self.__VERTICES_DTYPE))

        if xyz.dtype != self.__VERTICES_DTYPE:
            raise TypeError(
                f"Array of 'vertices' must be of dtype = {self.__VERTICES_DTYPE}"
            )

        if len(xyz) != 4:
            raise ValueError("Array of 'vertices' must be of shape (4, 3).")

        self._vertices = xyz

        if self.on_file:
            self.workspace.update_attribute(self, "vertices")

    @property
    def u_count(self) -> int:
        """
        Number of pixels in the u direction.
        :return: the number of pixels in the u direction.
        """
        return self.default_vertices[1, 0].astype(np.int32)

    @property
    def v_count(self) -> int:
        """
        Number of pixels in the v direction.
        :return: the number of pixels in the v direction.
        """
        return self.default_vertices[0, 1].astype(np.int32)

    @property
    def u_cell_size(self) -> float:
        """
        Cell size in the u direction.
        :return: the cell size in the u direction.
        """
        distance_u = np.linalg.norm(self.vertices[2] - self.vertices[3])
        return distance_u / self.u_count

    @property
    def v_cell_size(self) -> float:
        """
        Cell size in the v direction.
        :return: the cell size in the v direction.
        """
        distance_v = np.linalg.norm(self.vertices[0] - self.vertices[3])
        return distance_v / self.v_count

    def _create_geoimage_from_attributes(
        self, parent: None | Entity | Workspace = None, **kwargs
    ) -> GeoImage:
        """
        Create a new GeoImage from attributes.

        :param kwargs: The attributes to update.
        :param parent: the parent workspace or entity
            or a group containing the object.

        :return: a new instance of GeoImage.
        """
        if parent:
            if hasattr(parent, "h5file"):
                workspace = parent
            else:
                workspace = parent.workspace
                kwargs["parent"] = parent
        else:
            workspace = self.workspace

        new_attributes = GeoImageConversion.verify_kwargs(self, **kwargs)

        return GeoImage.create(workspace, **new_attributes)

    @staticmethod
    def _parse_tie_points(tie_points: tuple | list | np.ndarray) -> np.ndarray:
        """
        Iterate through the ModelTiepointTag to extract the tiepoints.

        Each tie point requires 6 values each:
            - 3 for the pixel location (i, j, k)
            - 3 for the world location (x, y, z)
        In the tag, it's stored as a flat list of values.

        :param tie_points: The ModelTiepointTag from a tiff image.

        :return: An array of tiepoints of shape (n, 2, 3).
        """
        tie_points = np.asarray(tie_points, dtype=float)

        if tie_points.ndim == 1:
            if tie_points.size % 6 != 0:
                raise ValueError(
                    "ModelTiepointTag length must be a multiple of 6. "
                    f"Got length {tie_points.size}."
                )
            tie_points = tie_points.reshape(-1, 2, 3)
        elif tie_points.ndim != 3 or tie_points.shape[1:] != (2, 3):
            raise ValueError(
                "Tie points must have shape (6n,) or (n, 2, 3). "
                f"Got {tie_points.shape}."
            )

        # Remove exact duplicate (pixel, world) pairs
        pairs_unique = np.unique(tie_points.reshape(-1, 6), axis=0)

        pix = pairs_unique[:, :3]
        wrd = pairs_unique[:, 3:]

        if np.unique(pix, axis=0).shape[0] != pairs_unique.shape[0]:
            raise ValueError(
                "Inconsistent tie points: identical pixel coordinates map to "
                "multiple world coordinates."
            )

        if np.unique(wrd, axis=0).shape[0] != pairs_unique.shape[0]:
            raise ValueError(
                "Inconsistent tie points: identical world coordinates map to "
                "multiple pixel coordinates."
            )

        return pairs_unique.reshape(-1, 2, 3)

    def _compute_image_corners_from_1_tie_point(
        self, points: np.ndarray, u_cell_size: float, v_cell_size: float
    ):
        """
        Compute world coordinates of image corners using 1 tie point and cell sizes.

        Assumes axis-aligned rectangular mapping with known pixel sizes in world units.

        :param points: List of [[i, j, k], [x, y, z]] pairs (must have at least 1).
        :param u_cell_size: World units per pixel in the i (horizontal) direction.
        :param v_cell_size: World units per pixel in the j (vertical) direction.

        :return: Array of shape (4, 3) with corner world coordinates.
        """
        # Extract first tie point
        pix_ref = np.array(points[0][0][:2], dtype=np.float64)  # [i, j]
        wrd_ref = np.array(points[0][1], dtype=np.float64)  # [x, y, z]

        # Compute pixel offsets from reference point
        delta_pix = self.default_vertices[::-1, :2] - pix_ref

        # Convert to world offsets using cell sizes
        delta_wrd = delta_pix * [u_cell_size, -v_cell_size]

        # Add world offsets to reference point (z remains constant)
        corners_wrd = np.column_stack(
            [
                wrd_ref[0] + delta_wrd[:, 0],  # x
                wrd_ref[1] + delta_wrd[:, 1],  # y
                np.full(4, wrd_ref[2]),  # z (constant)
            ]
        )

        return corners_wrd

    def _compute_image_corners_from_2_tie_points(
        self, points: np.ndarray, u_cell_size: float, v_cell_size: float
    ) -> np.ndarray | list[str]:
        # pylint: disable=too-many-locals
        """
        Compute world coordinates of image corners from 2 tie points,
        assuming orthogonality and known cell sizes.

        :param points: List of [[i, j, k], [x, y, z]] pairs.
        :param u_cell_size: World units per pixel in the i (horizontal) direction.
        :param v_cell_size: World units per pixel in the j (vertical) direction.

        :return: Array of shape (4, 3) with corner world coordinates, or list of error messages.
        """
        errors = []

        # Extract tie points
        pix0, pix1 = points[:2, 0, :2].astype(np.float64)
        wrd0, wrd1 = points[:2, 1, :].astype(np.float64)

        # Compute displacements
        delta_pix = pix1 - pix0
        delta_wrd = wrd1 - wrd0

        di, dj = delta_pix

        # Validate tie points differ in both directions
        if abs(di) < 1e-12 or abs(dj) < 1e-12:
            errors.append(
                f"Tie points must differ in both pixel directions. "
                f"Got di={di}, dj={dj}."
            )

        delta_wrd_norm = np.linalg.norm(delta_wrd)

        if delta_wrd_norm < 1e-12:
            errors.append("Tie points map to the same world coordinates")

        # Check consistency: |delta_wrd|² = (di*u_cell_size)² + (dj*v_cell_size)²
        expected_mag_sq = (di * u_cell_size) ** 2 + (dj * v_cell_size) ** 2
        if not np.isclose(expected_mag_sq, delta_wrd_norm**2, rtol=1e-3):
            errors.append(
                f"Tie points inconsistent with cell sizes. "
                f"Expected displacement magnitude: {np.sqrt(expected_mag_sq):.4f}, "
                f"Actual: {delta_wrd_norm:.4f}"
            )

        if errors:
            return errors

        # Build orthonormal basis in the plane
        e1 = delta_wrd / delta_wrd_norm

        # Find perpendicular direction
        if abs(e1[2]) < 0.9:
            reference = np.array([0.0, 0.0, 1.0])
        else:
            reference = np.array([1.0, 0.0, 0.0])

        e2 = np.cross(e1, reference)
        e2 = e2 / np.linalg.norm(e2)

        # Solve for rotation angle
        theta = np.arctan2(-dj * v_cell_size, di * u_cell_size)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Construct orthogonal basis vectors
        u = u_cell_size * (cos_theta * e1 + sin_theta * e2)
        v = v_cell_size * (-sin_theta * e1 + cos_theta * e2)

        # Compute origin and corners
        origin = wrd0 - pix0[0] * u - pix0[1] * v
        corners_pix = self.default_vertices[::-1, :2].astype(np.float64)

        corners_wrd = np.array(
            [origin + pix[0] * u + pix[1] * v for pix in corners_pix]
        )

        return corners_wrd

    def _compute_image_corners_from_3_tie_points(
        self, points: np.ndarray
    ) -> np.ndarray:
        """
        Compute world coordinates of image corners using 3 tie points.

        :param points: List of [[i, j, k], [x, y, z]] pairs.

        :return: Array of shape (4, 3) with corner world coordinates.
        """
        # Extract first 3 tie points
        pix = np.array([tp[0][:2] for tp in points[:3]], dtype=np.float64)  # (i, j)
        wrd = np.array([tp[1] for tp in points[:3]], dtype=np.float64)  # (x, y, z)

        # Build affine transformation: [i, j, 1] @ transform = [x, y, z]
        design_matrix = np.column_stack([pix, np.ones(3)])
        affine_transform = np.linalg.solve(design_matrix, wrd)

        corner_design_matrix = np.column_stack(
            [self.default_vertices[::-1, :2], np.ones(4)]
        )
        corners_wrd = corner_design_matrix @ affine_transform

        return corners_wrd

    def _compute_image_corners(
        self,
        points: np.ndarray,
        u_cell_size: float | None,
        v_cell_size: float | None,
        tol: float = 1e-4,
    ) -> np.ndarray | list[str]:
        """
        Compute world coordinates of image corners using tie points.

        :param points: List of [[i, j, k], [x, y, z]] pairs.
        :param u_cell_size: World units per pixel in the i (horizontal) direction.
        :param v_cell_size: World units per pixel in the j (vertical) direction.
        :param tol: Tolerance for coplanarity and affine consistency checks.

        :return:
            - Array of shape (4, 3) with corner world coordinates.
            - or list of error messages if validation fails.
        """
        errors = []
        if not are_coplanar(points[:, 1, :], tol):
            errors.append("Tie points are not coplanar; cannot compute image corners.")

        if not are_affine(points, tol):
            errors.append(
                "Tie points are not consistent with an affine transformation."
            )

        if points.shape[0] < 1:
            errors.append("At least 1 tie point is required to compute image corners.")

        if errors:
            return errors

        if points.shape[0] >= 3:
            corners = self._compute_image_corners_from_3_tie_points(points)

            # check cell size is consistent
            calc_u_cell_size = np.linalg.norm(corners[1] - corners[0]) / self.u_count
            calc_v_cell_size = np.linalg.norm(corners[0] - corners[3]) / self.v_count

            # should never happen
            if (
                u_cell_size is not None
                and v_cell_size is not None
                and (
                    not np.isclose(calc_u_cell_size, u_cell_size, rtol=tol)
                    or not np.isclose(calc_v_cell_size, v_cell_size, rtol=tol)
                )
            ):
                return [
                    "Computed cell sizes from tie points do not match provided cell sizes.\n"
                    f"Computed: ({calc_u_cell_size}, {calc_v_cell_size}), "
                    f"Provided: ({u_cell_size}, {v_cell_size})"
                ]

            return corners

        if u_cell_size is None or v_cell_size is None:
            return [
                "Cell sizes must be provided when only 1 or 2 tie points are available."
            ]
        if points.shape[0] == 1:
            return self._compute_image_corners_from_1_tie_point(
                points, u_cell_size, v_cell_size
            )

        return self._compute_image_corners_from_2_tie_points(
            points, u_cell_size, v_cell_size
        )
