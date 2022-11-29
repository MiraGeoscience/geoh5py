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

from io import BytesIO

import numpy as np
from PIL import Image

from .. import objects


class Convert:
    def __init__(self, entity):
        """
        Converter class from an object to another.
        WARNING: just GeoImage to Grid2d and Grid2D to GeoImage are implemented yet.
        :param entity: the entity to convert
        """

        self._entity = entity

    def geoimage_to_grid2d(
        self,
        transform: str = "GRAY",
        **grid2d_kwargs,
    ) -> objects.Grid2D:
        """
        Create a geoh5py :obj:geoh5py.objects.grid2d.Grid2D from the geoimage in the same workspace.
        :param transform: the type of transform ; if "GRAY" convert the image to grayscale ;
        if "RGB" every band is sent to a data of a grid.
        :param **grid2d_kwargs: Any argument supported by :obj:`geoh5py.objects.grid2d.Grid2D`.
        :return: the new created Grid2D.
        """
        if not isinstance(self._entity, objects.GeoImage):
            raise TypeError(
                f"Entity must be a 'GeoImage', {type(self._entity)} passed instead"
            )
        if transform not in ["GRAY", "RGB"]:
            raise KeyError(
                f"'transform' has to be 'GRAY' or 'RGB', you entered {transform} instead."
            )
        if self._entity.vertices is None:
            raise AttributeError("The 'vertices' has to be previously defined")

        # define name and elevation
        name = grid2d_kwargs.get("name", self._entity.name)
        elevation = grid2d_kwargs.get("elevation", 0)
        workspace = grid2d_kwargs.get("workspace", self._entity.workspace)
        grid2d_kwargs.pop("workspace", None)

        # get geographic information
        u_origin = self._entity.vertices[0, 0]
        v_origin = self._entity.vertices[2, 1]
        u_count = self._entity.default_vertices[1, 0]
        v_count = self._entity.default_vertices[0, 1]
        u_cell_size = abs(u_origin - self._entity.vertices[1, 0]) / u_count
        v_cell_size = abs(v_origin - self._entity.vertices[0, 1]) / v_count

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
        value = Image.open(BytesIO(self._entity.image_data.values))
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

    def grid_to_tag(self) -> dict:
        """
        Compute the tag dictionary of the Grid2D as required by the
        :obj:geoh5py.objects.geo_image.GeoImage object.
        :return: a dictionary of the tag.
        """
        if not isinstance(self._entity, objects.Grid2D):
            raise TypeError(
                f"Entity must be 'Grid2D', {type(self._entity)} passed instead"
            )
        if not isinstance(self._entity.v_cell_size, np.ndarray) or not isinstance(
            self._entity.u_cell_size, np.ndarray
        ):
            raise AttributeError("The Grid2D has no geographic information")

        if self._entity.u_count is None or self._entity.v_count is None:
            raise AttributeError("The Grid2D has no number of cells")

        u_origin, v_origin, z_origin = self._entity.origin.item()
        v_oposite = v_origin + self._entity.v_cell_size * self._entity.v_count
        tag = {
            256: (self._entity.u_count,),
            257: (self._entity.v_count,),
            33550: (
                self._entity.u_cell_size[0],
                self._entity.v_cell_size[0],
                0.0,
            ),
            33922: (
                0.0,
                0.0,
                0.0,
                u_origin,
                v_oposite,
                z_origin,
            ),
        }

        return tag

    def data_to_pil_format(self, data_name: str) -> np.array:
        """
        Convert a data of the Grid2D to a numpy array
        with a format compatible with :obj:`PIL.Image` object.
        :param data_name: the name of the data to extract.
        :return: the data formatted with the right shape,
        beetweem 0 and 255, as uint8.
        """
        if not isinstance(self._entity, objects.Grid2D):
            raise TypeError(
                f"Entity must be 'Grid2D', {type(self._entity)} passed instead"
            )

        # get the data
        data = self._entity.get_data(data_name)[0].values

        # reshape them
        data = data.reshape(self._entity.v_count, self._entity.u_count)[::-1]

        # normalize them #todo: change the no number value in np.nan
        min_, max_ = np.nanmin(data), np.nanmax(data)
        data = (data - min_) / (max_ - min_)
        data *= 255

        return data.astype(np.uint8)

    def grid2d_to_geoimage(
        self, data_list: list | str, **geoimage_kwargs
    ) -> objects.GeoImage:
        """
        Create a :obj:geoh5py.objects.geo_image.GeoImage object from the current Grid2D.
        :param data_list: the list of the data name to pass as band in the image.
        The len of the list can only be 1, 3, 4.
        :param **geoimage_kwargs: any argument of :obj:`geoh5py.objects.geo_image.GeoImage`.
        :return: a new georeferenced :obj:`geoh5py.objects.geo_image.GeoImage`.
        """
        if not isinstance(self._entity, objects.Grid2D):
            raise TypeError(
                f"Entity must be 'Grid2D', {type(self._entity)} passed instead"
            )
        # catch exception if str is entered instead of a list
        if isinstance(data_list, str):
            data_list = [data_list]

        # prepare the image
        if isinstance(data_list, list):
            if len(data_list) not in [1, 3, 4]:
                raise IndexError("Only 1, 3, or 4 layers can be selected")

            data = np.empty((self._entity.v_count, self._entity.u_count, 0)).astype(
                np.uint8
            )
            for data_name in data_list:
                if data_name not in self._entity.get_data_list():
                    raise KeyError(
                        f"The key {data_list} is not in the data: {self._entity.get_data_list()}"
                    )

                data_temp = self.data_to_pil_format(data_name)
                data = np.dstack((data, data_temp))
        else:
            raise TypeError(
                "The type of the keys must be a string or a list,",
                f"but you entered an {type(data_list)} ",
            )

        # convert to PIL
        if data.shape[-1] == 1:
            image = Image.fromarray(data[:, :, 0], mode="L")
        elif data.shape[-1] == 3:
            image = Image.fromarray(data, mode="RGB")
        elif data.shape[-1] == 4:
            image = Image.fromarray(data, mode="CMYK")

        workspace = geoimage_kwargs.get("workspace", self._entity.workspace)
        geoimage_kwargs.pop("workspace", None)

        # create a geoimage
        new_geoimage = objects.GeoImage.create(
            workspace, image=image, tag=self.grid_to_tag(), **geoimage_kwargs
        )

        # georeference it
        new_geoimage.georeferencing_from_tiff()

        return new_geoimage
