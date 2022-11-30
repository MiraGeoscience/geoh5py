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

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from ... import objects
from .base_conversion import ConversionBase

if TYPE_CHECKING:
    from ...objects import GeoImage, Grid2D


class Grid2dToGeoImage(ConversionBase):
    """
    Convert a :obj:geoh5py.objects.grid2d.Grid2D object
    to a georeferenced :obj:geoh5py.objects.geo_image.GeoImage object.
    """

    def __init__(self, entity: Grid2D):
        """
        :param entity: the :obj:geoh5py.objects.grid2d.Grid2D to convert.
        """
        if not isinstance(entity, objects.Grid2D):
            raise TypeError(f"Entity must be 'Grid2D', {type(entity)} passed instead")

        super().__init__(entity)
        self.entity: Grid2D

    def grid_to_tag(self) -> dict:
        """
        Compute the tag dictionary of the Grid2D as required by the
        :obj:geoh5py.objects.geo_image.GeoImage object.
        :return: a dictionary of the tag.
        """
        if not isinstance(self.entity.v_cell_size, np.ndarray) or not isinstance(
            self.entity.u_cell_size, np.ndarray
        ):
            raise AttributeError("The Grid2D has no geographic information")

        if self.entity.u_count is None or self.entity.v_count is None:
            raise AttributeError("The Grid2D has no number of cells")

        u_origin, v_origin, z_origin = self.entity.origin.item()
        v_oposite = v_origin + self.entity.v_cell_size * self.entity.v_count
        tag = {
            256: (self.entity.u_count,),
            257: (self.entity.v_count,),
            33550: (
                self.entity.u_cell_size[0],
                self.entity.v_cell_size[0],
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
        # get the data
        data = self.entity.get_data(data_name)[0].values

        # reshape them
        data = data.reshape(self.entity.v_count, self.entity.u_count)[::-1]

        # normalize them #todo: change the no number value in np.nan
        min_, max_ = np.nanmin(data), np.nanmax(data)
        data = (data - min_) / (max_ - min_)
        data *= 255

        return data.astype(np.uint8)

    def __call__(self, data_list: list | str, **geoimage_kwargs) -> GeoImage:
        """
        Create a :obj:geoh5py.objects.geo_image.GeoImage object from the current Grid2D.
        :param data_list: the list of the data name to pass as band in the image.
        The len of the list can only be 1, 3, 4.
        :param **geoimage_kwargs: any argument of :obj:`geoh5py.objects.geo_image.GeoImage`.
        :return: a new georeferenced :obj:`geoh5py.objects.geo_image.GeoImage`.
        """
        # catch exception if str is entered instead of a list
        if isinstance(data_list, str):
            data_list = [data_list]

        # prepare the image
        if isinstance(data_list, list):
            if len(data_list) not in [1, 3, 4]:
                raise IndexError("Only 1, 3, or 4 layers can be selected")

            data = np.empty((self.entity.v_count, self.entity.u_count, 0)).astype(
                np.uint8
            )
            for data_name in data_list:
                if data_name not in self.entity.get_data_list():
                    raise KeyError(
                        f"The key {data_list} is not in the data: {self.entity.get_data_list()}"
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

        workspace = geoimage_kwargs.get("workspace", self.entity.workspace)
        geoimage_kwargs.pop("workspace", None)

        # create a geoimage
        new_geoimage = objects.GeoImage.create(
            workspace, image=image, tag=self.grid_to_tag(), **geoimage_kwargs
        )

        # georeference it
        new_geoimage.georeferencing_from_tiff()

        return new_geoimage
