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
        grid2d_attributes["origin"] = np.asarray(
            tuple(input_entity.vertices[3]),
            dtype=[("x", float), ("y", float), ("z", float)],
        )

        grid2d_attributes["u_count"] = input_entity.default_vertices[1, 0]
        grid2d_attributes["v_count"] = input_entity.default_vertices[0, 1]

        # define the points
        point1 = np.array([input_entity.vertices[3, 0], input_entity.vertices[3, 1]])
        point2 = np.array([input_entity.vertices[2, 0], input_entity.vertices[2, 1]])
        point3 = np.array([input_entity.vertices[0, 0], input_entity.vertices[0, 1]])

        # Compute the distances
        distance_u = np.linalg.norm(point2 - point1)
        distance_v = np.linalg.norm(point3 - point1)

        # Now compute the cell sizes
        grid2d_attributes["u_cell_size"] = distance_u / grid2d_attributes["u_count"]
        grid2d_attributes["v_cell_size"] = distance_v / grid2d_attributes["v_count"]

        grid2d_attributes["elevation"] = grid2d_attributes.get("elevation", 0)

        if input_entity.rotation is not None:
            grid2d_attributes["rotation"] = input_entity.rotation

        return grid2d_attributes

    @classmethod
    def add_gray_data(cls, values: np.ndarray, output: Grid2D, name: str):
        """
        Send the image as gray in the new :obj:'geoh5py.objects.grid2d.Grid2D'.
        :param values: Input image values as an array of int.
        :param output: the new :obj:'geoh5py.objects.grid2d.Grid2D'.
        :param name: the name of the new :obj:'geoh5py.objects.grid2d.Grid2D'.
        """
        if values.ndim > 1:
            values = np.asarray(Image.fromarray(values).convert("L"))

        output.add_data(
            data={
                f"{name}_0": {
                    "values": values.astype(np.uint32)[::-1],
                    "association": "CELL",
                }
            }
        )

    @classmethod
    def add_color_data(cls, values: np.ndarray, output: Grid2D, name: str):
        """
        Send the image as rgb in the new :obj:'geoh5py.objects.grid2d.Grid2D'.
        :param values: Input image values as an array of int.
        :param output: the new :obj:'geoh5py.objects.grid2d.Grid2D'.
        :param name: the name of the new :obj:'geoh5py.objects.grid2d.Grid2D'.
        """
        if values.ndim != 3:
            raise IndexError("To export to color image, the array must be 3d.")

        for ind in range(values.shape[2]):
            output.add_data(
                {
                    f"{name}_{ind}": {
                        "values": values.astype(np.uint32)[::-1, :, ind],
                        "association": "CELL",
                    }
                }
            )

    @classmethod
    def add_data_2dgrid(
        cls, input_entity: GeoImage, output: Grid2D, transform: str | None, name: str
    ):
        """
        Select the type of the image transformation.
        :param input_entity: :obj:'geoh5py.objects.geo_image.GeoImage' object.
        :param output: the new :obj:'geoh5py.objects.grid2d.Grid2D'.
        :param transform: the type of the image transformation.
        :param name: the name of the new :obj:'geoh5py.objects.grid2d.Grid2D'.
        """
        if transform is not None and transform not in ["GRAY"]:
            raise ValueError(f"Transform can only be 'GRAY', not {transform}.")
        # add the data to the 2dgrid
        values = np.asarray(Image.open(BytesIO(input_entity.image_data.values)))

        if values.ndim == 2 or transform == "GRAY":
            cls.add_gray_data(values, output, name)
        else:
            cls.add_color_data(values, output, name)

    @classmethod
    def to_grid2d(
        cls,
        input_entity: GeoImage,
        transform: str | None,
        copy_children=True,
        **grid2d_kwargs,
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
