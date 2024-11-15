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

from typing import cast

import numpy as np

from ...data import NumericData
from ...objects import DrapeModel
from ...workspace import Workspace
from .base import BaseMerger


class DrapeModelMerger(BaseMerger):
    _type: type = DrapeModel

    @classmethod
    def create_object(
        cls, workspace: Workspace, input_entities: list, **kwargs
    ) -> DrapeModel:
        """
        Create a new DrapeModel from a list of input DrapeModels.

        It merges the cells together and create a new object with the merged cells.

        :param workspace: The workspace to create the object in.
        :param input_entities: The list of input entities to merge together.
        :param kwargs: The kwargs to pass to the object creation.

        :return: The newly created object merged from input_entities.
        """

        layers: list = []
        prisms: list = []
        previous_prism: int = 0
        previous_layer: int = 0
        ghost_prism: np.ndarray = np.array([])
        ghost_layer: np.ndarray = np.array([])

        for input_entity in input_entities:
            # get the values of the entity
            temp_prisms: np.ndarray = input_entity.prisms
            temp_layers: np.ndarray = input_entity.layers

            if len(temp_prisms) < 2:
                raise ValueError("All DrapeModel entities must have at least 2 prisms.")

            # get the first ghost
            if ghost_prism.size > 0:
                # append last ghost prism and layer
                prisms.append(ghost_prism)
                layers.append(ghost_layer)

                # create the first ghost point
                ghost_prism, ghost_layer = cls._ghost_point(
                    temp_prisms[0],
                    temp_prisms[1],
                    previous_prism - 1,
                    previous_layer - 1,
                )

                # append first ghost prism and layer
                prisms.append(ghost_prism)
                layers.append(ghost_layer)

            # add the entity prisms and layers to the list
            temp_prisms[:, -2] += previous_prism
            temp_layers[:, 0] += previous_layer

            previous_prism = temp_prisms[-1, -1] + temp_prisms[-1, -2] + 2
            previous_layer = temp_layers[-1, 0] + 3

            prisms.append(temp_prisms)
            layers.append(temp_layers)

            # create a last ghost prism and layer
            ghost_prism, ghost_layer = cls._ghost_point(
                temp_prisms[-1], temp_prisms[-2], previous_prism - 2, previous_layer - 2
            )

        # create an object
        output = cls._type.create(  # type: ignore
            workspace, prisms=np.vstack(prisms), layers=np.vstack(layers), **kwargs
        )

        return output

    @staticmethod
    def _ghost_point(
        point: np.ndarray,
        mirror: np.ndarray,
        previous_prism: int,
        previous_layer: int,
    ) -> np.ndarray | np.ndarray:
        """
        Create a ghost point prism and layer based on two points.
        :param point: The point to create the ghost from.
        :param mirror: The mirror point to create the ghost from.
        :param previous_prism: The ID of the previous prism.
        :param previous_layer: The ID of the previous layer.
        :return: A prism and a layer for the ghost point.
        """

        # create a mirrored point for last layer
        ghost_prism: np.ndarray = point.copy()
        ghost_layer: np.ndarray = np.empty(3)
        ghost_prism[:3] = 2 * point[:3] - mirror[:3]

        ghost_prism[3] = previous_prism
        ghost_prism[4] = 1

        ghost_layer[0] = previous_layer
        ghost_layer[1] = 0
        ghost_layer[2] = ghost_prism[2]

        return np.expand_dims(ghost_prism, 0), np.expand_dims(ghost_layer, 0)

    @classmethod
    def merge_data(
        cls,
        out_entity,
        input_entities: list[DrapeModel],
    ):
        super().merge_data(out_entity, input_entities)

        ind_map: list = []
        data_count: int = 0
        n_values: int = out_entity.n_cells - (len(input_entities) - 1) * 2
        for ghost, input_entity in enumerate(input_entities):
            if input_entity.n_cells is None:
                continue
            ind_map += [
                np.arange(data_count, data_count + input_entity.n_cells),
                [n_values + ghost * 2, n_values + ghost * 2 + 1],
            ]
            data_count += input_entity.n_cells

        # get all the values in the output entity
        for data in out_entity.children:
            if not isinstance(data, NumericData) or data.values is None:
                continue
            data.values = data.values[np.hstack(ind_map[:-1])]

    @classmethod
    def validate_structure(cls, input_entity: DrapeModel):
        """
        Validate the input entity structure and raises error if incompatible.
        :param input_entity: the input entity to validate.
        """
        # verify if the input entity have prism and layers
        if not (
            isinstance(cast(DrapeModel, input_entity).prisms, np.ndarray)
            and isinstance(cast(DrapeModel, input_entity).layers, np.ndarray)
        ):
            raise AttributeError("All entities must have prisms and layers.")
