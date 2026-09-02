# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2026 Mira Geoscience Ltd.                                     '
#                                                                              '
#  This file is part of geoh5py.                                               '
#                                                                              '
#  geoh5py is free software: you can redistribute it and/or modify             '
#  it under the terms of the GNU Lesser General Public License as published by '
#  the Free Software Foundation, either version 3 of the License, or           '
#  (at your option) any later version.                                         '
#                                                                              '
#  geoh5py is distributed in the hope that it will be useful,                  '
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              '
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               '
#  GNU Lesser General Public License for more details.                         '
#                                                                              '
#  You should have received a copy of the GNU Lesser General Public License    '
#  along with geoh5py.  If not, see <https://www.gnu.org/licenses/>.           '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

from .arrays import ArraySource, CallableArraySource, LazyArray
from .entity import Attributes, PydanticEntity
from .entity_type import DataType, EntityType, GroupType, ObjectType
from .points import VERTICES_DTYPE, PointsModel
from .project import ProjectAttributes
from .root import ROOT_TYPE_UID, Root, RootAttributes, RootType
from .serialization import Geoh5EntityPayload, Geoh5Writer
from .workspace import DEFAULT_PAGE_SIZE, Workspace


# Allow callers to write "from geoh5py_pydantic import xyz" instead of importing from submodules.
__all__ = [
    "DEFAULT_PAGE_SIZE",
    "ROOT_TYPE_UID",
    "VERTICES_DTYPE",
    "ArraySource",
    "Attributes",
    "CallableArraySource",
    "DataType",
    "EntityType",
    "Geoh5EntityPayload",
    "Geoh5Writer",
    "GroupType",
    "LazyArray",
    "ObjectType",
    "PointsModel",
    "ProjectAttributes",
    "PydanticEntity",
    "Root",
    "RootAttributes",
    "RootType",
    "Workspace",
]
