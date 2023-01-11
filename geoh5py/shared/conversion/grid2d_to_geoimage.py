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

from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np
from PIL import Image

from ... import objects
from ...data import Data
from ...shared import FLOAT_NDV
from .base_conversion import ConversionBase

if TYPE_CHECKING:
    from ...objects import Grid2D


class Grid2dToGeoImage(ConversionBase):
    """
    Convert a :obj:'geoh5py.objects.grid2d.Grid2D' object
    to a georeferenced :obj:'geoh5py.objects.geo_image.GeoImage' object.
    """

    def __init__(self, entity: Grid2D):
        """
        :param entity: the :obj:'geoh5py.objects.grid2d.Grid2D' to convert.
        """
        if not isinstance(entity, objects.Grid2D):
            raise TypeError(f"Entity must be 'Grid2D', {type(entity)} passed instead")

        super().__init__(entity)
        self._data = None
        self._image = None
        self._tag: dict | None = None
        self.entity: Grid2D

    def grid_to_tag(self):
        """
        Compute the tag dictionary of the Grid2D as required by the
        :obj:'geoh5py.objects.geo_image.GeoImage' object.
        """
        if not isinstance(self.entity.v_cell_size, np.ndarray) or not isinstance(
            self.entity.u_cell_size, np.ndarray
        ):
            raise AttributeError("The Grid2D has no geographic information")

        if self.entity.u_count is None or self.entity.v_count is None:
            raise AttributeError("The Grid2D has no number of cells")

        u_origin, v_origin, z_origin = self.entity.origin.item()
        v_oposite = v_origin + self.entity.v_cell_size * self.entity.v_count

        self._tag = {
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

    def data_to_pil_format(self, data: np.array) -> np.array:
        """
        Convert a numpy array with a format compatible with :obj:`PIL.Image` object.
        :param data: the data to convert.
        :return: the data formatted with the right shape,
        between 0 and 255, as uint8.
        """
        # reshape them
        data = data.reshape(self.entity.v_count, self.entity.u_count)[::-1]

        # remove nan values
        data = np.where(data == FLOAT_NDV, np.nan, data)

        # normalize them
        min_, max_ = np.nanmin(data), np.nanmax(data)
        data = (data - min_) / (max_ - min_)
        data *= 255

        return data.astype(np.uint8)

    def key_to_data(self, key: str | int | UUID | Data) -> np.array:
        """
        Extract the data from the entity in :obj:'np.array' format;
        The data can be of type: ':obj:str', ':obj:int', ':obj:UUID', or ':obj:Data'.
        :param key: the key of the data to extract.
        :return: an np. array containing the data.
        """
        # get the values
        if isinstance(key, str):
            data = self.entity.get_data(key)
        elif isinstance(key, int):
            if key > len(self.entity.get_entity_list()):
                raise IndexError(
                    "'int' values pass as key can't be larger than number of data,",
                    f"data number: {len(self.entity.get_entity_list())}, key: {key}",
                )
            key_ = self.entity.get_entity_list()[key]
            data = self.entity.get_entity(key_)  # type: ignore
        elif isinstance(key, UUID):
            data = self.entity.get_entity(key)  # type: ignore
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

    def data_from_keys(self, keys: list | str | int | UUID | Data):
        """
        Take a list of (or a unique) key to extract from the object,
        and create a :obj:'np.array' with those data.
        :param keys: the list of the data to extract.
        """

        # if unique key transform to list
        if isinstance(keys, (str, int, UUID, Data)):
            keys = [keys]

        # prepare the image
        if isinstance(keys, list):
            self._data = np.empty((self.entity.v_count, self.entity.u_count, 0)).astype(
                np.uint8
            )

            for key in keys:
                data_temp = self.key_to_data(key)
                data_temp = self.data_to_pil_format(data_temp)
                self._data = np.dstack((self._data, data_temp))
        else:
            raise TypeError(
                "The keys must be pass as a list",
                f"but you entered a {type(keys)} ",
            )

    def convert_to_pillow(self):
        """
        Convert the data from :obj:'np.array' to :obj:'PIL.Image' format.
        """
        if not isinstance(self._data, np.ndarray):
            raise AttributeError("No data is selected.")

        if self._data.shape[-1] == 1:
            self._image = Image.fromarray(self._data[:, :, 0], mode="L")
        elif self._data.shape[-1] == 3:
            self._image = Image.fromarray(self._data, mode="RGB")
        elif self._data.shape[-1] == 4:
            self._image = Image.fromarray(self._data, mode="CMYK")
        else:
            raise IndexError("Only 1, 3, or 4 layers can be selected")

    def verify_kwargs(self, **geoimage_kwargs):
        """
        Verify if the kwargs are valid.
        :param geoimage_kwargs: the kwargs to verify.
        """
        super().verify_kwargs()

        self.name = geoimage_kwargs.get("name", self.entity.name)
        self.change_workspace_parent(**geoimage_kwargs)

    def get_attributes(self, **kwargs):
        """
        Get the attributes of the entity.
        In order to select the layers, you can pass "keys" in the kwargs
        (see 'data_from_keys()' function).
        By default the first data layer is selected.
        :param kwargs: the kwargs to pass to the function.
        """
        super().get_attributes()

        # get the tag of the data
        self.grid_to_tag()

        # get the data
        self.data_from_keys(kwargs.get("keys", 0))
        self.convert_to_pillow()

    def create_output(self, **kwargs):
        """
        Create the output of the object.
        :param kwargs: the kwargs to pass to the :obj:'geoh5py.objects.geo_image.GeoImage'.
        """
        super().create_output()

        # create a geoimage
        self._output = objects.GeoImage.create(
            self.workspace_output, image=self._image, tag=self._tag, **kwargs
        )

    def add_data_output(self, **_):
        """
        Add the data to the workspace (georeference the data).
        """
        # pylint: disable=unused-argument
        super().add_data_output()

        # georeference it
        self._output.georeferencing_from_tiff()
