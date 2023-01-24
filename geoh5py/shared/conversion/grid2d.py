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

from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np
from PIL import Image

from ... import objects
from ...data import Data
from ...shared import FLOAT_NDV
from .base import CellObject

if TYPE_CHECKING:
    from ...objects import GeoImage, Grid2D


class Grid2DConversion(CellObject):
    """
    Convert a :obj:'geoh5py.objects.grid2d.Grid2D' object
    to a georeferenced :obj:'geoh5py.objects.geo_image.GeoImage' object.
    """

    # convert to geoimage

    @classmethod
    def grid_to_tag(cls, input_entity: Grid2D) -> dict:
        """
        Compute the tag dictionary of the :obj:'geoh5py.objects.geo_image.Grid2D'
        as required by the :obj:'geoh5py.objects.geo_image.GeoImage' object.
        :param input_entity: the Grid2D object to convert.
        :return tag: the tag dictionary.
        """
        if not isinstance(input_entity.v_cell_size, np.ndarray) or not isinstance(
            input_entity.u_cell_size, np.ndarray
        ):
            raise AttributeError("The Grid2D has no geographic information")

        if input_entity.u_count is None or input_entity.v_count is None:
            raise AttributeError("The Grid2D has no number of cells")

        if not isinstance(input_entity.origin, np.ndarray):
            raise AttributeError("The Grid2D has no origin")

        u_origin, v_origin, z_origin = input_entity.origin.tolist()
        v_oposite = v_origin + input_entity.v_cell_size * input_entity.v_count

        tag = {
            256: (input_entity.u_count,),
            257: (input_entity.v_count,),
            33550: (
                input_entity.u_cell_size[0],
                input_entity.v_cell_size[0],
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

    @classmethod
    def data_to_pil_format(cls, input_entity: Grid2D, data: np.ndarray) -> np.ndarray:
        """
        Convert a numpy array with a format compatible with :obj:`PIL.Image` object.
        :param input_entity: the Grid2D object to convert.
        :param data: the data to convert.
        :return: the data formatted with the right shape,
        between 0 and 255, as uint8.
        """
        # reshape them
        data = data.reshape(input_entity.v_count, input_entity.u_count)[::-1]

        # remove nan values
        data = np.where(data == FLOAT_NDV, np.nan, data)

        # normalize them
        min_, max_ = np.nanmin(data), np.nanmax(data)
        data = (data - min_) / (max_ - min_)
        data *= 255

        return data.astype(np.uint8)

    @classmethod
    def key_to_data(
        cls, input_entity: Grid2D, key: str | int | UUID | Data
    ) -> np.ndarray:
        """
        Extract the data from the entity in :obj:'np.array' format;
        The data can be of type: ':obj:str', ':obj:int', ':obj:UUID', or ':obj:Data'.
        :param input_entity: the Grid2D object to convert.
        :param key: the key of the data to extract.
        :return: an np. array containing the data.
        """
        # get the values
        if isinstance(key, str):
            data = input_entity.get_data(key)
        elif isinstance(key, int):
            if key > len(input_entity.get_entity_list()):
                raise IndexError(
                    "'int' values pass as key can't be larger than number of data,",
                    f"data number: {len(input_entity.get_entity_list())}, key: {key}",
                )
            key_ = input_entity.get_entity_list()[key]
            data = input_entity.get_entity(key_)  # type: ignore
        elif isinstance(key, UUID):
            data = input_entity.get_entity(key)  # type: ignore
        elif isinstance(key, Data):
            data = key  # type: ignore
        else:
            raise TypeError(
                "The dtype of the keys must be :",
                "'str', 'int', 'Data', 'UUID',",
                f"but you entered a {type(key)}",
            )

        # verify if data exists
        if isinstance(data, list):
            if len(data) != 1:
                raise KeyError(f"the key '{key}' you entered does not exists.")
            data = data[0]  # type: ignore

        return data.values  # type: ignore

    @classmethod
    def data_from_keys(
        cls, input_entity: Grid2D, keys: list | str | int | UUID | Data
    ) -> np.ndarray:
        """
        Take a list of (or a unique) key to extract from the object,
        and create a :obj:'np.array' with those data.
        :param input_entity: the Grid2D object to convert.
        :param keys: the list of the data to extract.
        :return data: the data extracted from the object.
        """

        # if unique key transform to list
        if isinstance(keys, (str, int, UUID, Data)):
            keys = [keys]

        # prepare the image
        if isinstance(keys, list):
            data = np.empty((input_entity.v_count, input_entity.u_count, 0)).astype(
                np.uint8
            )

            for key in keys:
                data_temp = cls.key_to_data(input_entity, key)
                data_temp = cls.data_to_pil_format(input_entity, data_temp)
                data = np.dstack((data, data_temp))

            return data

        raise TypeError(
            "The keys must be pass as a list",
            f"but you entered a {type(keys)} ",
        )

    @classmethod
    def convert_to_pillow(cls, data: np.ndarray) -> Image:
        """
        Convert the data from :obj:'np.array' to :obj:'PIL.Image' format.
        """
        if not isinstance(data, np.ndarray):
            raise AttributeError("No data is selected.")

        if data.shape[-1] == 1:
            return Image.fromarray(data[:, :, 0], mode="L")
        if data.shape[-1] == 3:
            return Image.fromarray(data, mode="RGB")
        if data.shape[-1] == 4:
            return Image.fromarray(data, mode="CMYK")

        raise IndexError("Only 1, 3, or 4 layers can be selected")

    @classmethod
    def to_geoimage(
        cls,
        input_entity: Grid2D,
        keys: list | str | int | UUID | Data,
        **geoimage_kwargs,
    ) -> GeoImage:
        """
        Convert the object to a :obj:'GeoImage' object.
        :param input_entity: the Grid2D object to convert.
        :param keys: the data to extract.
        :param geoimage_kwargs: the kwargs to pass to the :obj:'GeoImage' object.
        """

        properties = cls.verify_kwargs(input_entity, **geoimage_kwargs)

        # get the tag of the data
        geoimage_kwargs["tag"] = cls.grid_to_tag(input_entity)

        # get the data
        data = cls.data_from_keys(input_entity, keys)
        geoimage_kwargs["image"] = cls.convert_to_pillow(data)

        # create a geoimage
        output = objects.GeoImage.create(properties["workspace"], **geoimage_kwargs)

        # georeference it
        output.georeferencing_from_tiff()

        # copy properties
        cls.copy_properties(input_entity, output, **geoimage_kwargs)

        return output
