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

import numpy as np

from ...objects import Points
from ...workspace import Workspace
from .base import BaseMerger


class PointsMerger(BaseMerger):
    _type: type = Points

    @classmethod
    def create_object(
        cls, workspace: Workspace, input_entities: list, **kwargs
    ) -> Points:
        # create the vertices
        vertices = np.vstack([input_entity.vertices for input_entity in input_entities])

        # create an object of type
        output = cls._type.create(  # type: ignore
            workspace, vertices=vertices, **kwargs
        )

        return output

    @classmethod
    def validate_structure(cls, input_entity: Points):
        """
        Validate the input entity structure and raises error if incompatible.
        :param input_entity: the input entity to validate.
        """
        # verify if the input entity have vertices
        if not isinstance(input_entity.vertices, np.ndarray):
            raise AttributeError("All entities must have vertices.")
