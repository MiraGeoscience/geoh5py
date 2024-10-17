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

from ...objects import CellObject, Curve, Surface
from ...workspace import Workspace
from .points import PointsMerger


class CellMerger(PointsMerger):
    _type: type = CellObject

    @classmethod
    def create_object(
        cls, workspace: Workspace, input_entities: list, **kwargs
    ) -> CellObject:
        """
        Create a new object of type cls._type from a list of input entities.
        It merges the cells together and create a new object with the merged cells.

        :param workspace: The workspace to create the object in.
        :param input_entities: The list of input entities to merge together.
        :param kwargs: The kwargs to pass to the object creation.

        :return: The newly created object merged from input_entities.
        """
        # create the vertices
        vertices = np.vstack([input_entity.vertices for input_entity in input_entities])

        # merge the simplices
        cells: list = []
        previous: int = 0
        for entity in input_entities:
            temp_cells = entity.cells + previous
            cells.append(temp_cells)
            previous = np.nanmax(temp_cells) + 1

        # create an object of type
        output = cls._type.create(  # type: ignore
            workspace, vertices=vertices, cells=np.vstack(cells).tolist(), **kwargs
        )

        return output


class CurveMerger(CellMerger):
    _type = Curve


class SurfaceMerger(CellMerger):
    _type = Surface
