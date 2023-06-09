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
from .data_type import DataType
from .text_data import TextData

PARAMETERS = ["Colour"]
ATTRIBUTES = ["tag", "text", "attrib"]


class VisualParameters(TextData):
    _xml: ET.Element | None = None
    _association = DataAssociationEnum.OBJECT

    def __init__(
        self,
        data_type: DataType,
        **kwargs,
    ):
        if (
            not isinstance(data_type, DataType)
            or data_type.primitive_type != self.primitive_type()
        ):
            raise TypeError(
                "Input 'data_type' must be a DataType object of primitive_type 'TEXT'."
            )

        data_type.name = "XmlData"
        data_type.description = "XML format text data"

        super().__init__(data_type, **kwargs)

        if self.entity_type.name == "Entity":
            self.entity_type.name = self.name

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
    def colour(self) -> None | list:
        """
        Colour of the object in [Alpha, Red, Green, Blue] format.

        Each value is an integer between 0 and 255.
        The colour value is stored as a single integer converted from
        a byte string of the form 'BBGGRR00' where 'BB' is the blue value,
        'GG' is the green value, 'RR' is the red converted from hexadecimal format.
        """
        element = self.get_tag("Colour")

        if element is None or not element.text:
            return None

        c_string = (int(element.text)).to_bytes(4, byteorder="little").hex()

        return [int(c_string[i : i + 2], 16) for i in range(0, 8, 2)]

    @colour.setter
    def colour(self, argb: list | tuple | np.ndarray):
        if (
            not isinstance(argb, (list, tuple, np.ndarray))
            or len(argb) != 4
            or not all(isinstance(val, int) for val in argb)
        ):
            raise TypeError("Input 'colour' values must be a list of 4 integers.")

        byte_string = ""
        for val in argb:
            byte_string += f"{val:02x}"

        value = int.from_bytes(bytes.fromhex(byte_string), "little")

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
