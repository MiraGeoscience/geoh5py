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

import xml.etree.ElementTree as ET

import numpy as np

from .data_association_enum import DataAssociationEnum
from .text_data import TextData

PARAMETERS = ["Colour"]
ATTRIBUTES = ["tag", "text", "attrib"]


class VisualParameters(TextData):
    _xml: ET.Element | None = None
    _association = DataAssociationEnum.OBJECT

    @property
    def xml(self) -> ET.Element:
        """
        :obj:`str` XML string.
        """
        if self._xml is None:
            if isinstance(self.values, str):
                str_xml = self.values
            else:
                str_xml = """
                    <IParameterList Version="1.0">
                    </IParameterList>
                """

            self._xml = ET.fromstring(str_xml)

        return self._xml

    @property
    def values(self):
        if self._xml is not None:
            self._values = ET.tostring(self._xml, encoding="unicode")

        elif (getattr(self, "_values", None) is None) and self.on_file:
            values = self.workspace.fetch_values(self)
            if isinstance(values, (np.ndarray, str, type(None))):
                self._values = values

        return self._values

    @values.setter
    def values(self, values) -> int | None:
        raise UserWarning(
            "Cannot set values for VisualParameters. Set supported attributes instead."
        )

    @property
    def colour(self) -> None | str:
        """
        :obj:`dict` the Colour dictionary.
        """
        element = self.get_tag("Colour")
        return getattr(element, "text", None)

    @colour.setter
    def colour(self, argb: list | tuple | np.ndarray):
        if (
            not isinstance(argb, (list, tuple, np.ndarray))
            or len(argb) != 4
            or not all(isinstance(val, int) for val in argb)
        ):
            raise TypeError("Input 'colour' values must be a list of 4 integers.")

        byte_string = b""
        for val in argb:
            byte_string += bytes.fromhex(hex(val)[2:])

        value = int.from_bytes(byte_string, "little")

        self.set_tag("Colour", str(value))

    def get_tag(self, tag: str) -> None | ET.Element:
        """
        Check if the tag is a valid parameter.
        :param tag: the key of the tag to check.
        :return: True if the tag is a valid parameter.
        """
        element = self.xml.find(tag)
        return element  # type: ignore

    def set_tag(self, tag: str, value: str):
        """
        Check if the tag is a valid parameter.
        :param tag: the key of the tag to check.
        :param value: the value to set.
        """

        if not isinstance(value, str):
            raise TypeError(
                f"Input 'value' for VisualParameters.{tag} must be of type {str}."
            )

        if self.xml.find(tag) is None:
            ET.SubElement(self.xml, tag)

        element = self.get_tag(tag)

        if element is not None:
            element.text = value
            self.workspace.update_attribute(self, "values")
