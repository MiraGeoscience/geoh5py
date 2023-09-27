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

import numpy as np

from ...objects import ObjectBase, Surface
from ...workspace import Workspace
from .base import BaseMerger


class SurfaceMerger(BaseMerger):
    _type = Surface

    @classmethod
    def validate_type(cls, input_entity):
        # want to make sure that the input entities are Points, no subclasses
        if type(input_entity) is not cls._type:  # pylint: disable=unidiomatic-typecheck
            raise TypeError("The input entities must be a list of geoh5py Surfaces.")

    @classmethod
    def create_object(
        cls, workspace: Workspace, input_entities: list[ObjectBase], **kwargs
    ) -> Surface:
        # create the vertices
        vertices = np.vstack([input_entity.vertices for input_entity in input_entities])

        # merge the simplices
        simplices: list = []
        previous: int = 0
        for entity in input_entities:
            temp_simplices = entity.cells + previous
            simplices.append(temp_simplices)
            previous = np.nanmax(temp_simplices) + 1

        # create an object of type
        output = cls._type.create(
            workspace, vertices=vertices, cells=np.vstack(simplices).tolist(), **kwargs
        )

        return output
