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

from io import BytesIO
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from ... import objects
from .base import BaseConversion

if TYPE_CHECKING:
    from ...objects import GeoImage, Grid2D


class GeoImageConversion(BaseConversion):
    """
    Convert a :obj:'geoh5py.objects.geo_image.GeoImage' object.
    """

    @classmethod
    def convert_to_grid2d_reference(
        cls, input_entity: GeoImage, grid2d_attributes
    ) -> dict:
        """
        Extract the geographic information from the entity.
        """
        if input_entity.vertices is None:
            raise AttributeError("GeoImage has no vertices")

        # get geographic information
        grid2d_attributes["u_origin"] = input_entity.vertices[0, 0]
        grid2d_attributes["v_origin"] = input_entity.vertices[2, 1]
        grid2d_attributes["u_count"] = input_entity.default_vertices[1, 0]
        grid2d_attributes["v_count"] = input_entity.default_vertices[0, 1]

        grid2d_attributes["u_cell_size"] = (
            abs(grid2d_attributes["u_origin"] - input_entity.vertices[1, 0])
            / grid2d_attributes["u_count"]
        )
        grid2d_attributes["v_cell_size"] = (
            abs(grid2d_attributes["v_origin"] - input_entity.vertices[0, 1])
            / grid2d_attributes["v_count"]
        )
        grid2d_attributes["elevation"] = grid2d_attributes.get("elevation", 0)
        return grid2d_attributes

    @classmethod
    def add_gray_data(cls, input_entity: GeoImage, output: Grid2D, name: str):
        """
        Send the image as gray in the new :obj:'geoh5py.objects.grid2d.Grid2D'.
        :param input_entity: :obj:'geoh5py.objects.geo_image.GeoImage' object.
        :param output: the new :obj:'geoh5py.objects.grid2d.Grid2D'.
        :param name: the name of the new :obj:'geoh5py.objects.grid2d.Grid2D'.
        """
        value = Image.open(BytesIO(input_entity.image_data.values))

        output.add_data(
            data={
                f"{name}_GRAY": {
                    "values": np.array(value.convert("L")).astype(np.uint32)[::-1],
                    "association": "CELL",
                }
            }
        )

    @classmethod
    def add_rgb_data(cls, input_entity: GeoImage, output: Grid2D, name: str):
        """
        Send the image as rgb in the new :obj:'geoh5py.objects.grid2d.Grid2D'.
        :param input_entity: :obj:'geoh5py.objects.geo_image.GeoImage' object.
        :param output: the new :obj:'geoh5py.objects.grid2d.Grid2D'.
        :param name: the name of the new :obj:'geoh5py.objects.grid2d.Grid2D'.
        """
        value = Image.open(BytesIO(input_entity.image_data.values))

        if np.array(value).shape[-1] != 3:
            raise IndexError("To export to RGB the image has to have 3 bands")

        output.add_data(
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

    @classmethod
    def add_cmyk_data(cls, input_entity: GeoImage, output: Grid2D, name: str):
        """
        Send the image as cmyk in the new :obj:'geoh5py.objects.grid2d.Grid2D'.
        :param input_entity: :obj:'geoh5py.objects.geo_image.GeoImage' object.
        :param output: the new :obj:'geoh5py.objects.grid2d.Grid2D'.
        :param name: the name of the new :obj:'geoh5py.objects.grid2d.Grid2D'.
        """
        value = Image.open(BytesIO(input_entity.image_data.values))

        if np.array(value).shape[-1] != 4:
            raise IndexError("To export to CMYK the image has to have 4 bands")

        output.add_data(
            data={
                f"{name}_C": {
                    "values": np.array(value).astype(np.uint32)[::-1, :, 0],
                    "association": "CELL",
                },
                f"{name}_M": {
                    "values": np.array(value).astype(np.uint32)[::-1, :, 1],
                    "association": "CELL",
                },
                f"{name}_Y": {
                    "values": np.array(value).astype(np.uint32)[::-1, :, 2],
                    "association": "CELL",
                },
                f"{name}_K": {
                    "values": np.array(value).astype(np.uint32)[::-1, :, 3],
                    "association": "CELL",
                },
            }
        )

    @classmethod
    def add_data_2dgrid(
        cls, input_entity: GeoImage, output: Grid2D, transform: str, name: str
    ):
        """
        Select the type of the image transformation.
        :param input_entity: :obj:'geoh5py.objects.geo_image.GeoImage' object.
        :param output: the new :obj:'geoh5py.objects.grid2d.Grid2D'.
        :param transform: the type of the image transformation.
        :param name: the name of the new :obj:'geoh5py.objects.grid2d.Grid2D'.
        """
        # add the data to the 2dgrid
        if transform == "GRAY":
            cls.add_gray_data(input_entity, output, name)
        elif transform == "RGB":
            cls.add_rgb_data(input_entity, output, name)
        elif transform == "CMYK":
            cls.add_cmyk_data(input_entity, output, name)
        else:
            raise KeyError(
                f"'transform' has to be 'GRAY', 'CMYK' or 'RGB', you entered {transform} instead."
            )

    @classmethod
    def to_grid2d(
        cls, input_entity: GeoImage, transform: str, copy_children=True, **grid2d_kwargs
    ) -> Grid2D:
        """
        Transform the :obj:'geoh5py.objects.image.Image' to a :obj:'geoh5py.objects.grid2d.Grid2D'.
        :param input_entity: :obj:'geoh5py.objects.geo_image.GeoImage' object.
        :param transform: the transforming type option.
        :return: the new :obj:'geoh5py.objects.grid2d.Grid2D'.
        """
        workspace = cls.validate_workspace(input_entity, **grid2d_kwargs)
        grid2d_kwargs = cls.verify_kwargs(input_entity, **grid2d_kwargs)
        grid2d_kwargs = cls.convert_to_grid2d_reference(input_entity, grid2d_kwargs)
        output = objects.Grid2D.create(
            workspace,
            **grid2d_kwargs,
        )

        if copy_children:
            cls.add_data_2dgrid(input_entity, output, transform, output.name)

        return output
