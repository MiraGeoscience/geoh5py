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

# pylint: disable=too-few-public-methods

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from ..data import Data, DataAssociationEnum, FloatData, NumericData


class PropertyGroupType(ABC):
    """
    Class to define the basic structure of a property group type.
    """

    no_modify: bool = False

    @classmethod
    @abstractmethod
    def verify(cls, children: list[Data]):
        """
        Verify that the children are of the correct type for this group type

        If unsuccessful, raise a ValueError

        :param children: The children to verify
        """


class SimpleType(PropertyGroupType):
    @classmethod
    def verify(cls, children: list[Data]):
        """
        Accept any children
        """


class DepthType(PropertyGroupType):
    @classmethod
    def verify(cls, children: list[Data]):
        if (
            len(children) < 1
            or not children[0].association == DataAssociationEnum.DEPTH
            or not isinstance(children[0], FloatData)
        ):
            raise TypeError(
                "First children of 'Depth table' property group type "
                "must be a FloatData of 'Depth' association. "
            )


class DipDirType(PropertyGroupType):
    no_modify = True

    @classmethod
    def verify(cls, children: list[Data]):
        if len(children) != 2 or not all(
            isinstance(child, NumericData) for child in children
        ):
            raise TypeError(
                "Children of 'Dip direction & dip' property group type "
                "must be a list of 2 NumericData entities"
            )


class IntervalType(PropertyGroupType):
    @classmethod
    def verify(cls, children: list[Data]):
        if (
            len(children) < 2
            or not children[0].association == DataAssociationEnum.DEPTH
            or not all(isinstance(child, FloatData) for child in children[:2])
        ):
            raise TypeError(
                "First two children of 'Interval table' property group type "
                "must be FloatData of 'Interval' association."
            )


class MultiElementType(PropertyGroupType):
    @classmethod
    def verify(cls, children: list[Data]):
        if not all(isinstance(child, NumericData) for child in children):
            raise TypeError(
                "Children of 'Multi-element' property group type "
                "must be a list of NumericData entities"
            )


class StrikeDipType(PropertyGroupType):
    no_modify = True

    @classmethod
    def verify(cls, children: list[Data]):
        if len(children) != 2 or not all(
            isinstance(child, NumericData) for child in children
        ):
            raise TypeError(
                "Children of 'Strike & dip' property group type "
                "must be a list of 2 NumericData entities"
            )


class VectorType(PropertyGroupType):
    no_modify = True

    @classmethod
    def verify(cls, children: list[Data]):
        if len(children) != 3 or not all(
            isinstance(child, NumericData) for child in children
        ):
            raise TypeError(
                "Children of '3D vector' property group type "
                "must be a list of 3 NumericData entities"
            )


GROUP_TYPES = {
    "Depth table": DepthType,
    "Dip direction & dip": DipDirType,
    "Interval table": IntervalType,
    "Multi-element": MultiElementType,
    "Simple": SimpleType,
    "Strike & dip": StrikeDipType,
    "3D vector": VectorType,
}


class GroupTypeEnum(str, Enum):
    """
    Supported property group types.
    """

    DEPTH = "Depth table"
    DIPDIR = "Dip direction & dip"
    INTERVAL = "Interval table"
    MULTI = "Multi-element"
    SIMPLE = "Simple"
    STRIKEDIP = "Strike & dip"
    VECTOR = "3D vector"

    @classmethod
    def find_type(cls, data: list[Data]):
        """
        Determine the group type based on the data.
        """
        if all(isinstance(d, NumericData) for d in data):
            return cls.MULTI
        return cls.SIMPLE

    def verify(self, data: list[Data]):
        """
        Validate the data based on the group type.
        """
        GROUP_TYPES[self.value].verify(data)

    @property
    def no_modify(self) -> bool:
        """
        Get the name of the group type.
        """
        return GROUP_TYPES[self.value].no_modify
