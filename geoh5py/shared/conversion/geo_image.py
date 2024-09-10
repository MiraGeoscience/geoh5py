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

    @staticmethod
    def convert_to_grid2d_reference(geoimage: GeoImage, grid2d_attributes) -> dict:
        """
        Extract the geographic information from the entity.
        """
        if geoimage.vertices is None or geoimage.default_vertices is None:
            raise AttributeError("GeoImage has no vertices.")

        # get geographic information
        grid2d_attributes["origin"] = np.asarray(
            tuple(geoimage.vertices[3]),
            dtype=[("x", float), ("y", float), ("z", float)],
        )

        grid2d_attributes["u_count"] = geoimage.default_vertices[1, 0].astype(np.int32)
        grid2d_attributes["v_count"] = geoimage.default_vertices[0, 1].astype(np.int32)

        # Compute the distances
        distance_u = np.linalg.norm(geoimage.vertices[2] - geoimage.vertices[3])
        distance_v = np.linalg.norm(geoimage.vertices[0] - geoimage.vertices[3])

        # Now compute the cell sizes
        grid2d_attributes["u_cell_size"] = distance_u / grid2d_attributes["u_count"]
        grid2d_attributes["v_cell_size"] = distance_v / grid2d_attributes["v_count"]

        grid2d_attributes["elevation"] = grid2d_attributes.get("elevation", 0)

        if geoimage.rotation is not None:
            grid2d_attributes["rotation"] = geoimage.rotation

        if geoimage.dip is not None:
            grid2d_attributes["dip"] = geoimage.dip

        return grid2d_attributes

    @staticmethod
    def add_gray_data(values: np.ndarray, output: Grid2D):
        """
        Send the image as gray in the new :obj:'geoh5py.objects.grid2d.Grid2D'.
        :param values: Input image values as an array of int.
        :param output: the new :obj:'geoh5py.objects.grid2d.Grid2D'.
        """
        if values.ndim != 2:
            raise ValueError("To export to gray image, the array must be 2d. ")

        output.add_data(
            data={
                "band[0]": {
                    "values": values[::-1].flatten(),
                    "association": "CELL",
                }
            }
        )

    @staticmethod
    def add_color_data(values: np.ndarray, output: Grid2D):
        """
        Send the image color bands to data.

        :param values: Input image values as an array of int.
        :param output: the new :obj:'geoh5py.objects.grid2d.Grid2D'.
        """
        if values.ndim != 3:
            raise IndexError("To export to color image, the array must be 3d.")

        for ind in range(values.shape[2]):
            output.add_data(
                {
                    f"band[{ind}]": {
                        "values": values[::-1, :, ind].flatten(),
                        "association": "CELL",
                    }
                }
            )

    @staticmethod
    def add_data_2dgrid(geoimage: Image, output: Grid2D):
        """
        Select the type of the image transformation.
        :param geoimage: :obj:'geoh5py.objects.geo_image.GeoImage' object.
        :param output: the new :obj:'geoh5py.objects.grid2d.Grid2D'.
        """
        values = np.asarray(geoimage)

        if values.ndim == 2:
            GeoImageConversion.add_gray_data(values, output)
        else:
            GeoImageConversion.add_color_data(values, output)

    @staticmethod
    def to_grid2d(
        geoimage: GeoImage,
        mode: str | None,
        copy_children=True,
        **grid2d_kwargs,
    ) -> Grid2D:
        """
        Transform the :obj:'geoh5py.objects.image.Image' to a :obj:'geoh5py.objects.grid2d.Grid2D'.
        :param geoimage: :obj:'geoh5py.objects.geo_image.GeoImage' object.
        :param mode: The outgoing image mode option.

        :return: the new :obj:'geoh5py.objects.grid2d.Grid2D'.
        """

        workspace, grid2d_kwargs = GeoImageConversion.validate_workspace(
            geoimage, **grid2d_kwargs
        )
        grid2d_kwargs = GeoImageConversion.verify_kwargs(geoimage, **grid2d_kwargs)
        grid2d_kwargs = GeoImageConversion.convert_to_grid2d_reference(
            geoimage, grid2d_kwargs
        )

        output = objects.Grid2D.create(
            workspace,
            **grid2d_kwargs,
        )

        if geoimage.image is not None:
            image = geoimage.image.copy()

            if mode is not None and mode != image.mode:
                image = image.convert(mode if mode != "GRAY" else "L")

            if copy_children:
                GeoImageConversion.add_data_2dgrid(image, output)

        return output
