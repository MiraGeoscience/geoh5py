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

from typing import cast

import numpy as np

from ...objects import DrapeModel, ObjectBase
from ...workspace import Workspace
from .base import BaseMerger


class DrapeModelMerger(BaseMerger):
    _type: type = DrapeModel

    @classmethod
    def create_object(
        cls, workspace: Workspace, input_entities: list[ObjectBase], **kwargs
    ) -> DrapeModel:
        """
        Create a new object of type cls._type from a list of input entities.
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
        for input_entity in input_entities:
            temp_prisms: np.ndarray = cast(
                np.ndarray, cast(DrapeModel, input_entity).prisms
            )
            temp_layers: np.ndarray = cast(
                np.ndarray, cast(DrapeModel, input_entity).layers
            )

            temp_prisms[:, -2] += previous_prism
            temp_layers[:, 0] += previous_layer

            previous_prism = temp_prisms[-1, -1] + temp_prisms[-1, -2]
            previous_layer = temp_layers[-1, 0] + 1

            prisms.append(temp_prisms)
            layers.append(temp_layers)

        # create an object
        output = cls._type.create(  # type: ignore
            workspace, prisms=np.vstack(prisms), layers=np.vstack(layers), **kwargs
        )

        return output

    @classmethod
    def validate_objects(cls, input_entities: list[ObjectBase]):
        """
        Validate the input entities types and raises error if incompatible.

        :param input_entities: a list of :obj:geoh5py.objects.ObjectBase objects.
        """
        # assert input entities is a list of len superior to 1
        if not isinstance(input_entities, list):
            raise TypeError("The input entities must be a list of geoh5py objects.")

        # assert input entities is a list of len superior to 1
        if len(input_entities) < 2:
            raise ValueError("Need more than one object to merge.")

        # assert input entities are of the same type
        if not all(
            type(input_entity) is type(input_entities[0])
            for input_entity in input_entities
        ):
            raise TypeError("All objects must be of the same type.")

        # assert input entities are of the same type
        cls.validate_type(input_entities[0])

        # verify if the all input entities have vertices
        if not all(
            (
                isinstance(cast(DrapeModel, input_entity).prisms, np.ndarray)
                and isinstance(cast(DrapeModel, input_entity).layers, np.ndarray)
            )
            for input_entity in input_entities
        ):
            raise AttributeError("All entities must have vertices.")
