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

from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .drillholes_concatenator import DrillholesConcatenator


class DrillholesGroupTable(ABC):
    """
    A class to store the information of a PropertyGroup.

    :param parent: The parent DrillholesConcatenator.
    :param name: The name of the PropertyGroup to extract.
    """

    def __init__(self, parent: DrillholesConcatenator, name: str):
        if not hasattr(parent, "unique_property_group_names"):
            raise TypeError("The parent must be a Concatenator instance.")

        self._parent: DrillholesConcatenator = parent

        # ensure name is in parent's unique_property_group_names
        if name not in parent.unique_property_group_names:
            raise KeyError(
                f"The name '{name}' is not in the parent's unique_property_group_names."
            )

        self._name: str = name

    # get all the property groups in the concatenator
    def association_by_drillhole(
        self,
        names: tuple,
    ) -> dict:
        """
        Based on the first name of the input list,
        get for every object index and count of all the data in 'names'

        :param names: The names of the data to extract.

        :return: A dictionary with the object uuid and the index of all the data.
        """
        # ensure all the names are in the index
        if not all(name in self._parent.index for name in names):
            raise KeyError(f"Data '{names}' not found in concatenated data.")

        object_index: dict = {}
        for drillhole in self._parent.index[names[0]]["Object ID"]:
            object_index[drillhole] = {}
            for name in names:
                if drillhole in self._parent.index[name]["Object ID"]:
                    object_index[drillhole][name] = list(
                        self._parent.index[name][
                            self._parent.index[name]["Object ID"] == drillhole
                        ][0]
                    )[:2]
                else:
                    object_index[drillhole][name] = [0, 0]

        return object_index

    def index_by_drillholes(self) -> dict:
        """
        Get the index of the data of the PropertyGroup by drillholes.

        :return: The index of the data by drillholes.
        """
        association, names = self._parent.unique_property_group_names[self._name]
        index_by_drillholes = self.association_by_drillhole(association + names)

        return index_by_drillholes

    @property
    def association_names(self) -> tuple[str]:
        """
        The names of the association
        """
        return self._parent.unique_property_group_names[self._name][0]
