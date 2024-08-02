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
from uuid import UUID

import numpy as np
from PIL import Image

from ... import objects
from ...data import Data
from ...shared import FLOAT_NDV
from ...shared.utils import PILLOW_ARGUMENTS, xy_rotation_matrix, yz_rotation_matrix
from .base import CellObjectConversion


if TYPE_CHECKING:
    from ...objects import GeoImage, Grid2D


class Grid2DConversion(CellObjectConversion):
    """
    Convert a :obj:'geoh5py.objects.grid2d.Grid2D' object
    to a georeferenced :obj:'geoh5py.objects.geo_image.GeoImage' object.
    """

    # convert to geoimage

    @staticmethod
    def grid_to_tag(input_entity: Grid2D) -> dict:
        """
        Compute the tag dictionary of the :obj:'geoh5py.objects.geo_image.Grid2D'
        as required by the :obj:'geoh5py.objects.geo_image.GeoImage' object.
        :param input_entity: the Grid2D object to convert.
        :return tag: the tag dictionary.
        """
        if not isinstance(input_entity.origin, np.ndarray):
            raise AttributeError("The Grid2D has no origin.")

        if input_entity.v_cell_size is None or input_entity.u_cell_size is None:
            raise AttributeError("The Grid2D has no cell size information.")

        if input_entity.u_count is None or input_entity.v_count is None:
            raise AttributeError("The Grid2D has no number of cells.")

        u_origin, v_origin, z_origin = input_entity.origin.tolist()
        v_oposite = v_origin + input_entity.v_cell_size * input_entity.v_count

        tag = {
            256: (input_entity.u_count,),
            257: (input_entity.v_count,),
            33550: (
                input_entity.u_cell_size,
                input_entity.v_cell_size,
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

    @staticmethod
    def data_to_pil_format(
        input_entity: Grid2D, data: np.ndarray, normalize: bool = True
    ) -> np.ndarray:
        """
        Convert a numpy array with a format compatible with :obj:`PIL.Image` object.
        :param input_entity: the Grid2D object to convert.
        :param data: the data to convert.
        :param normalize: if True, the data will be normalized between 0 and 255.
        :return: the data formatted with the right shape,
        between 0 and 255, as uint8.
        """
        # reshape them
        data = data.reshape(input_entity.v_count, input_entity.u_count)[::-1]

        # remove nan values
        data = np.where(data == FLOAT_NDV, np.nan, data)

        # normalize them
        if normalize:
            min_, max_ = np.nanmin(data), np.nanmax(data)
            data = (data - min_) / (max_ - min_)
            data *= 255

        return data.astype(np.uint8)

    @staticmethod
    def key_to_data(input_entity: Grid2D, key: str | int | UUID | Data) -> np.ndarray:
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
                raise KeyError(
                    f"The key '{key}' you entered does not exists;"
                    f"Valid keys are: {input_entity.get_data_list()}."
                )
            data = data[0]  # type: ignore

        return data.values  # type: ignore

    @staticmethod
    def data_from_keys(
        input_entity: Grid2D,
        keys: list | str | int | UUID | Data,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Take a list of (or a unique) key to extract from the object,
        and create a :obj:'np.array' with those data.
        :param input_entity: the Grid2D object to convert.
        :param normalize: if True, the data will be normalized between 0 and 255.
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
                data_temp = Grid2DConversion.key_to_data(input_entity, key)

                data_temp = Grid2DConversion.data_to_pil_format(
                    input_entity, data_temp, normalize
                )

                data = np.dstack((data, data_temp))

            return data

        raise TypeError(
            "The keys must be pass as a list",
            f"but you entered a {type(keys)} ",
        )

    @staticmethod
    def convert_to_pillow(data: np.ndarray, mode: str | None = None) -> Image:
        """
        Convert the data from :obj:'np.array' to :obj:'PIL.Image' format.
        """
        if not isinstance(data, np.ndarray):
            raise AttributeError("No data is selected.")

        if mode is None:
            if data.shape[-1] == 1:
                mode = "L"
                data = data[:, :, 0]

            if data.shape[-1] == 3:
                mode = "RGB"
            elif data.shape[-1] == 4:
                mode = "RGBA"

            if mode is None:
                raise IndexError("Only 1, 3, or 4 layers can be selected")

        if mode not in PILLOW_ARGUMENTS:
            raise NotImplementedError(f"The mode {mode} is not actually supported.")

        return Image.fromarray(data, mode=mode)

    @staticmethod
    def compute_vertices(input_entity: Grid2D) -> np.ndarray:
        """
        Compute the vertices of the geoimage to create based on the
        properties of the grid and its angle.
        :param input_entity: The grid2D object to convert.
        :return: A numpy array of 4 points in x,y,z.
        """
        # get the length of the x and y axis
        if (
            input_entity.u_count is None
            or input_entity.v_count is None
            or input_entity.u_cell_size is None
            or input_entity.v_cell_size is None
        ):
            raise AttributeError(
                "The grid2D object must have the following properties"
                ": u_count, v_count, u_cell_size, v_cell_size"
            )

        length_u = input_entity.u_count * input_entity.u_cell_size
        length_v = input_entity.v_count * input_entity.v_cell_size

        # Define the corners
        corners = np.array(
            [[0, length_v, 0], [length_u, length_v, 0], [length_u, 0, 0], [0, 0, 0]]
        )

        # rotate by the dip
        dip_matrix = yz_rotation_matrix(np.deg2rad(input_entity.dip))
        dipped_corners = dip_matrix @ corners.T

        # rotate by the rotation
        rotation_matrix = xy_rotation_matrix(np.deg2rad(input_entity.rotation))
        rotated_corners = rotation_matrix @ dipped_corners

        # add the origin
        shifted_corners = rotated_corners.T + np.array(input_entity.origin.tolist())

        return shifted_corners

    @staticmethod
    def to_geoimage(
        input_entity: Grid2D,
        keys: list | str | int | UUID | Data,
        mode: str | None = None,
        normalize: bool = True,
        **geoimage_kwargs,
    ) -> GeoImage:
        """
        Convert the object to a :obj:'GeoImage' object.
        :param input_entity: the Grid2D object to convert.
        :param keys: the data to extract.
        :param mode: the mode of the image to create.
        :param normalize: if True, the data will be normalized between 0 and 255.
        :param geoimage_kwargs: the kwargs to pass to the :obj:'GeoImage' object.
        """
        workspace, geoimage_kwargs = Grid2DConversion.validate_workspace(
            input_entity, **geoimage_kwargs
        )

        geoimage_kwargs = Grid2DConversion.verify_kwargs(
            input_entity, **geoimage_kwargs
        )

        # get the tag of the data
        geoimage_kwargs["tag"] = Grid2DConversion.grid_to_tag(input_entity)

        # get the data
        data = Grid2DConversion.data_from_keys(input_entity, keys, normalize=normalize)

        # define the image
        geoimage_kwargs["image"] = Grid2DConversion.convert_to_pillow(data, mode=mode)

        # define the vertices
        geoimage_kwargs["vertices"] = Grid2DConversion.compute_vertices(input_entity)

        # create a geoimage
        output = objects.GeoImage.create(workspace, **geoimage_kwargs)

        return output
