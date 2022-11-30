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
from .base_conversion import ConversionBase

if TYPE_CHECKING:
    from ...objects import GeoImage, Grid2D


class GeoImagetoGrid2D(ConversionBase):
    """
    Convert a :obj:geoh5py.objects.geo_image.GeoImage object
    to a georeferenced :obj:geoh5py.objects.grid2d.Grid2D object.
    """

    def __init__(self, entity: GeoImage):
        """
        :param entity: the :obj:geoh5py.objects.geo_image.GeoImage to convert.
        """
        if not isinstance(entity, objects.GeoImage):
            raise TypeError(
                f"Entity must be a 'GeoImage', {type(entity)} passed instead"
            )

        super().__init__(entity)
        self.entity: GeoImage

    def __call__(
        self,
        transform: str = "GRAY",
        **grid2d_kwargs,
    ) -> Grid2D:
        """
        Create a geoh5py :obj:geoh5py.objects.grid2d.Grid2D from the geoimage in the same workspace.
        :param transform: the type of transform ; if "GRAY" convert the image to grayscale ;
        if "RGB" every band is sent to a data of a grid.
        :param **grid2d_kwargs: Any argument supported by :obj:`geoh5py.objects.grid2d.Grid2D`.
        :return: the new created Grid2D.
        """
        if transform not in ["GRAY", "RGB"]:
            raise KeyError(
                f"'transform' has to be 'GRAY' or 'RGB', you entered {transform} instead."
            )
        if self.entity.vertices is None:
            raise AttributeError("The 'vertices' has to be previously defined")

        # define name and elevation
        name = grid2d_kwargs.get("name", self.entity.name)
        elevation = grid2d_kwargs.get("elevation", 0)
        workspace = grid2d_kwargs.get("workspace", self.entity.workspace)
        grid2d_kwargs.pop("workspace", None)

        # get geographic information
        u_origin = self.entity.vertices[0, 0]
        v_origin = self.entity.vertices[2, 1]
        u_count = self.entity.default_vertices[1, 0]
        v_count = self.entity.default_vertices[0, 1]
        u_cell_size = abs(u_origin - self.entity.vertices[1, 0]) / u_count
        v_cell_size = abs(v_origin - self.entity.vertices[0, 1]) / v_count

        # create the 2dgrid
        grid = objects.Grid2D.create(
            workspace,
            origin=[u_origin, v_origin, elevation],
            u_cell_size=u_cell_size,
            v_cell_size=v_cell_size,
            u_count=u_count,
            v_count=v_count,
            **grid2d_kwargs,
        )

        # add the data to the 2dgrid
        value = Image.open(BytesIO(self.entity.image_data.values))
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
