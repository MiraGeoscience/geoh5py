# #  Copyright (c) 2024 Mira Geoscience Ltd.
# #
# #  This file is part of geoh5py.
# #
# #  geoh5py is free software: you can redistribute it and/or modify
# #  it under the terms of the GNU Lesser General Public License as published by
# #  the Free Software Foundation, either version 3 of the License, or
# #  (at your option) any later version.
# #
# #  geoh5py is distributed in the hope that it will be useful,
# #  but WITHOUT ANY WARRANTY; without even the implied warranty of
# #  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# #  GNU Lesser General Public License for more details.
# #
# #  You should have received a copy of the GNU Lesser General Public License
# #  along with geoh5py.  If not, see <https://www.gnu.org/licenses/>.
#
from __future__ import annotations

from pydantic import BaseModel, model_validator
import uuid

from .base import Group

# from ..objects import Curve, Slicer


class InterpretationSection(BaseModel):
    """
    The class to store the interpretation section.

    :param normal_x: The normal x component.
    :param normal_y: The normal y component.
    :param normal_z: The normal z component.
    :param position_x: The position x component.
    :param position_y: The position y component.
    :param position_z: The position z component.
    """
    _attribute_map = {
        "Normal X": "normal_x",
        "Normal Y": "normal_y",
        "Normal Z": "normal_z",
        "Position X": "position_x",
        "Position Y": "position_y",
        "Position Z": "position_z",
    }

    normal_x: float
    normal_y: float
    normal_z: float
    position_x: float
    position_y: float
    position_z: float

    @model_validator(mode="before")
    @classmethod
    def map_attributes(cls, data: dict) -> dict:
        """
        Map the attributes to the right keys.

        :param data: The data to map.

        :return: The mapped data.
        """
        return {cls._attribute_map.get(key, key): value for key, value in data.items()}

    @property
    def map_back(self) -> dict:
        """
        Map the attributes back to the original keys of attribute map.

        :return: The mapped data.
        """
        return {key: getattr(self, value) for key, value in self._attribute_map.items()}


class InterpretationSection(Group):
    """The type for the basic Container group."""

    _attribute_map = {
        "Interpretation curves": "interpretation_curves",
        "Interpretation sections": "interpretation_sections",
        "Section object ID": "section_object_id"
    }

    _TYPE_UID = uuid.UUID(
        fields=(0x27EE59E1, 0xB1CE, 0x41EC, 0x8A, 0x86, 0x8BD3D229E198)
    )
    _default_name = "Interpretation Section"

    def __init__(self):
        super().__init__()

        self._interpretation_curve: None | list[Curve] = None
        self._interpretation_sections: None | list[InterpretationSection] = None

    # @property
    # def interpretation_curves(self) -> list[Curve] | None:
    #     """
    #     Get the interpretation curves.
    #
    #     :return: The interpretation curves.
    #     """
    #     return self._interpretation_curve
    #
    # @interpretation_curves.setter
    # def interpretation_curves(self, curves: list[Curve]):
    #     """
    #     Set the interpretation curves.
    #
    #     :param curves: The interpretation curves.
    #     """
    #     if not isinstance(curves, list):
    #         raise TypeError("The interpretation curves must be a list.")
    #
    #     if not all(isinstance(curve, Curve) and curve not in self.children for curve in curves):
    #         raise TypeError(
    #             "The interpretation curves must be a list of Curve"
    #             f"objects and a children of {self}."
    #         )
    #
    #     self._interpretation_curve = curves
    #
    # @property
    # def interpretation_sections(self) -> list[InterpretationSection] | None:
    #     """
    #     Get the interpretation sections.
    #
    #     :return: The interpretation sections.
    #     """
    #     return self._interpretation_sections
    #
    # @property
    # def section_object_id(self) -> str | None:
    #     """
    #     Get the section object ID.
    #
    #     :return: The section object ID.
    #     """
    #     return self.get_data(str)
