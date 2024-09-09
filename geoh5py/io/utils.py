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

from ..data import Data, DataType
from ..groups import Group, GroupType, PropertyGroup
from ..objects import ObjectBase, ObjectType
from ..shared.entity import Entity
from ..shared.entity_type import EntityType


def str_from_type(entity: Entity | PropertyGroup | EntityType) -> str:
    """
    Convert entity type to geoh5 string standard.

    :param entity: Entity, PropertyGroup or EntityType

    :return: String equivalent.
    """
    if isinstance(entity, Data):
        return "Data"

    if isinstance(entity, Group):
        return "Groups"

    if isinstance(entity, ObjectBase):
        return "Objects"

    if isinstance(entity, PropertyGroup):
        return "PropertyGroups"

    if isinstance(entity, EntityType):
        return "Types"

    raise TypeError(
        "Input value should be of type 'Data', 'Group' or 'ObjectBase'. "
        f"Provided {type(entity)}"
    )


def str_from_subtype(entity_type: EntityType) -> str:
    """
    Convert sub entity type to geoh5 string standard.

    :param entity_type: Sub-class of entity type.

    :return: String equivalent.
    """
    if isinstance(entity_type, DataType):
        return "Data types"
    if isinstance(entity_type, ObjectType):
        return "Object types"
    if isinstance(entity_type, GroupType):
        return "Group types"

    raise TypeError(
        "Input value should be of type 'Data', 'Group' or 'ObjectBase'. "
        f"Provided {type(entity_type)}"
    )
