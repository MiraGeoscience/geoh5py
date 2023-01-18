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

from io import BytesIO
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from ... import objects
from .base import ConversionBase

if TYPE_CHECKING:
    from ...objects import GeoImage, Grid2D


class GeoImageConversion(ConversionBase):
    """
    Convert a :obj:'geoh5py.objects.geo_image.GeoImage' object.
    """

    _entity: GeoImage

    def __init__(self, entity: GeoImage):
        """
        :param entity: the :obj:'geoh5py.objects.geo_image.GeoImage' to convert.
        """
        if not isinstance(entity, objects.GeoImage):
            raise TypeError(
                f"Entity must be a 'GeoImage', {type(entity)} passed instead"
            )

        super().__init__(entity)
        self._elevation = 0
        self._u_origin = None
        self._v_origin = None
        self._u_count = None
        self._v_count = None
        self._u_cell_size = None
        self._v_cell_size = None
        self.entity: GeoImage
        self.output: Grid2D
        self.name: str

    def convert_to_grid2d_reference(self):
        """
        Extract the geographic information from the entity.
        """
        if self.entity.vertices is None:
            raise AttributeError("GeoImage has no vertices")

        # get geographic information
        self._u_origin = self.entity.vertices[0, 0]
        self._v_origin = self.entity.vertices[2, 1]
        self._u_count = self.entity.default_vertices[1, 0]
        self._v_count = self.entity.default_vertices[0, 1]

        self._u_cell_size = (
            abs(self._u_origin - self.entity.vertices[1, 0]) / self._u_count
        )
        self._v_cell_size = (
            abs(self._v_origin - self.entity.vertices[0, 1]) / self._v_count
        )

    def add_gray_data(self):
        """
        Send the image as gray in the new :obj:'geoh5py.objects.grid2d.Grid2D'.
        """
        value = Image.open(BytesIO(self.entity.image_data.values))

        self.output.add_data(
            data={
                f"{self.name}_GRAY": {
                    "values": np.array(value.convert("L")).astype(np.uint32)[::-1],
                    "association": "CELL",
                }
            }
        )

    def add_rgb_data(self):
        """
        Send the image as rgb in the new :obj:'geoh5py.objects.grid2d.Grid2D'.
        """
        value = Image.open(BytesIO(self.entity.image_data.values))

        if np.array(value).shape[-1] != 3:
            raise IndexError("To export to RGB the image has to have 3 bands")

        self.output.add_data(
            data={
                f"{self.name}_R": {
                    "values": np.array(value).astype(np.uint32)[::-1, :, 0],
                    "association": "CELL",
                },
                f"{self.name}_G": {
                    "values": np.array(value).astype(np.uint32)[::-1, :, 1],
                    "association": "CELL",
                },
                f"{self.name}_B": {
                    "values": np.array(value).astype(np.uint32)[::-1, :, 2],
                    "association": "CELL",
                },
            }
        )

    def add_cmyk_data(self):
        """
        Send the image as cmyk in the new grid2d.
        """
        value = Image.open(BytesIO(self.entity.image_data.values))

        if np.array(value).shape[-1] != 4:
            raise IndexError("To export to CMYK the image has to have 4 bands")

        self.output.add_data(
            data={
                f"{self.name}_C": {
                    "values": np.array(value).astype(np.uint32)[::-1, :, 0],
                    "association": "CELL",
                },
                f"{self.name}_M": {
                    "values": np.array(value).astype(np.uint32)[::-1, :, 1],
                    "association": "CELL",
                },
                f"{self.name}_Y": {
                    "values": np.array(value).astype(np.uint32)[::-1, :, 2],
                    "association": "CELL",
                },
                f"{self.name}_K": {
                    "values": np.array(value).astype(np.uint32)[::-1, :, 3],
                    "association": "CELL",
                },
            }
        )

    def add_data_2dgrid(self, transform):
        """
        Select the type of the image transformation.
        :param transform: the transforming type option.
        """
        # add the data to the 2dgrid
        if transform == "GRAY":
            self.add_gray_data()
        elif transform == "RGB":
            self.add_rgb_data()
        elif transform == "CMYK":
            self.add_cmyk_data()
        else:
            raise KeyError(
                f"'transform' has to be 'GRAY', 'CMYK' or 'RGB', you entered {transform} instead."
            )

    def verify_kwargs(self, **grid2d_kwargs):
        """
        Verify if the kwargs are correct for the transformation;
        update the name, the elevation, and the workspace if needed.
        :param grid2d_kwargs:
        """
        self.name = grid2d_kwargs.get("name", self.entity.name)
        self._elevation = grid2d_kwargs.get("elevation", 0)
        self.change_workspace_parent(**grid2d_kwargs)

    def to_grid2d(self, transform: str, **grid2d_kwargs) -> Grid2D:
        """
        Transform the :obj:'geoh5py.objects.image.Image' to a :obj:'geoh5py.objects.grid2d.Grid2D'.
        :param transform: the transforming type option.
        :return: the new :obj:'geoh5py.objects.grid2d.Grid2D'.
        """

        self.verify_kwargs(**grid2d_kwargs)

        # get the vertices of the Grid2D
        self.convert_to_grid2d_reference()

        # create output object
        self.output = objects.Grid2D.create(
            self.workspace_output,
            origin=[self._u_origin, self._v_origin, self._elevation],
            u_cell_size=self._u_cell_size,
            v_cell_size=self._v_cell_size,
            u_count=self._u_count,
            v_count=self._v_count,
            **grid2d_kwargs,
        )

        # add the data to the Grid2D
        self.add_data_2dgrid(transform)

        # convert the properties of the geoimage to the grid
        self.copy_properties()

        return self.output
