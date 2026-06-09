# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2020-2026 Mira Geoscience Ltd.                                '
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

from uuid import UUID

import numpy as np

from geoh5py.groups.base import Group
from geoh5py.groups.property_group import PropertyGroup
from geoh5py.shared.entity import Entity
from geoh5py.shared.utils import (
    dict_mapper,
    entity2uuid,
    str2uuid,
)


class GIFtoolsGroup(Group):
    """The type for a GIFtools group."""

    _TYPE_UID = UUID(fields=(0x585B3218, 0xC24B, 0x41FE, 0xAD, 0x1F, 0x24D5E6E8348A))
    _default_name = "GIFtools Project"
    _attribute_map: dict = Group._attribute_map.copy()
    _attribute_map.update(
        {
            "giftoolsVersion": "version",
            "Can add group": "can_add_group",
        }
    )

    def __init__(
        self, gif_parameters: list[dict] | None = None, version: int = 1, **kwargs
    ):
        self._version: int = version
        self._can_add_group: bool = True

        super().__init__(**kwargs)

        self._gif_parameters = self._validate_parameters(gif_parameters)

    def add_children(
        self, children: Entity | PropertyGroup | list[Entity | PropertyGroup]
    ):
        """
        :param children: Add a list of entities as
            :obj:`~geoh5py.shared.entity.Entity.children`
        """
        super().add_children(children)
        for child in self.children:
            if hasattr(child, "parameters") and child.uid not in self._gif_parameters:
                form = child.parameters
                form["uuid"] = child.uid
                self._gif_parameters[child.uid] = form

        if self.on_file:
            self.workspace.update_attribute(self, "gif_parameters")

    @property
    def can_add_group(self) -> bool:
        """
        Attribute to determine if groups can be added.
        """
        return self._can_add_group

    @can_add_group.setter
    def can_add_group(self, value: bool) -> None:
        if not isinstance(value, bool | np.integer):
            raise TypeError(f"Input 'can_add_group' must be of type {bool}.")
        self._can_add_group = bool(value)

    @property
    def version(self) -> int:
        """
        GIFtools version number.
        """
        return self._version

    @version.setter
    @version.setter
    def version(self, version: int) -> None:
        if not isinstance(version, (int, np.integer)):
            raise TypeError(f"Input 'version' must be of type {int}.")

        self._version = int(version)

    @property
    def gif_parameters(self) -> list[dict]:
        """
        Metadata attached to the entity.

        Return a copy of the dictionary to avoid accidental modifications.
        """
        return list(self._gif_parameters.values())

    @staticmethod
    def _validate_parameters(value: list[dict] | None) -> dict[UUID, dict]:

        if value is None:
            value = []

        if not isinstance(value, list):
            raise TypeError(f"Input 'gif_parameters' must be of type {list}.")

        promoted = dict_mapper(value, [str2uuid, entity2uuid])
        dict_values: dict[UUID, dict] = {}
        for element in promoted:
            if not isinstance(element, dict) or "uuid" not in element:
                raise ValueError(
                    "Each gif_parameters entry must be a dict containing a 'uuid' key."
                )
            dict_values[element["uuid"]] = element

        return dict_values

    def update_child_parameters(self, child_uid: UUID, **kwargs):
        """
        Update an entry in the list of gif parameters.

        :param child_uid: UUID of the child entity.
        :param kwargs: Keyword arguments
        """
        form = self._gif_parameters[child_uid]
        self._gif_parameters[child_uid] = update_dict_parameters(form, **kwargs)
        self.workspace.update_attribute(self, "gif_parameters")


def update_dict_parameters(parameters: dict, **kwargs) -> dict:
    """
    Given a dictionary of values, update the entries with the provided keyword arguments.
    The keys of the keyword arguments must match the keys in the dictionary.

    :param parameters: Dictionary of values to update
    :param kwargs: Keyword arguments
    :return: Updated dictionary
    """
    dict_values = dict_mapper(kwargs, [str2uuid, entity2uuid])
    for key, value in dict_values.items():
        if key not in parameters:
            raise ValueError(f"Parameter '{key}' is not supported.")

        if "value" in parameters[key]:
            parameters[key]["value"] = value
        else:
            parameters[key] = value

    return parameters
