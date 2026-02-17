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

import xml.etree.ElementTree as ET
from collections.abc import Sequence

import numpy as np

from .data_association_enum import DataAssociationEnum
from .text_data import TextData


PARAMETERS = ["Colour"]
ATTRIBUTES = ["tag", "text", "attrib"]


class VisualParameters(TextData):
    _xml: ET.Element | None = None
    _association = DataAssociationEnum.OBJECT

    def __init__(
        self,
        association: DataAssociationEnum = DataAssociationEnum.OBJECT,
        public: bool = False,
        **kwargs,
    ):
        super().__init__(association=association, public=public, **kwargs)

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
    def values(self) -> str | None:
        if self._xml is not None:
            self._values = ET.tostring(self._xml, encoding="unicode")

        elif (getattr(self, "_values", None) is None) and self.on_file:
            values = self.workspace.fetch_values(self)
            if isinstance(values, (np.ndarray, str, type(None))):
                self._values = values

        return self._values

    @values.setter
    def values(self, values: np.ndarray | str | None):
        if not isinstance(values, (np.ndarray, str, type(None))):
            raise ValueError(
                f"Input 'values' for {self} must be of type {np.ndarray}  str or None."
            )

        self._values = values
        self.workspace.update_attribute(self, "values")

    @property
    def colour(self) -> None | list:
        """
        Colour of the object in [Red, Green, Blue] format.

        Each value is an integer between 0 and 255.
        The colour value is stored as a single integer converted from
        a byte string of the form 'RRGGBB' where 'BB' is the blue value,
        'GG' is the green value, 'RR' is the red converted from hexadecimal format.
        """
        element = self.get_tag("Colour")

        if element is None or not element.text:
            return None

        c_string = (int(element.text)).to_bytes(4, byteorder="little").hex()

        # Flip the colour order from BGR to RGB
        return [int(c_string[i : i + 2], 16) for i in range(0, 8, 2)][2::-1]

    @colour.setter
    def colour(self, rgb: Sequence):
        if (
            not isinstance(rgb, Sequence)
            or len(rgb) not in [3, 4]
            or not all(isinstance(val, int) for val in rgb)
        ):
            raise TypeError("Input 'colour' values must be a list of 3 or 4 integers.")

        rgb = list(rgb)
        if len(rgb) == 3:
            rgb += [255]

        # Flip the colour order from RGB to BGR
        rgb = rgb[2::-1] + [rgb[3]]

        byte_string = "".join(f"{val:02x}" for val in rgb)
        byte_string.join(f"{255:02x}")  # alpha value
        value = int.from_bytes(bytes.fromhex(byte_string), "little")

        self.set_tag("Colour", str(value))

    @property
    def filter_basement(self) -> None | float:
        """
        Filter basement tag for VPmodel.
        """
        element = self.get_tag("Filterbasement")

        if element is None or not element.text:
            return None

        return float(element.text)

    @filter_basement.setter
    def filter_basement(self, value: float):
        """
        Set the filter basement value.

        :param value: The filter basement value.
        """
        if not isinstance(value, (int, float)):
            raise TypeError("Input 'filter_basement' must be a number.")

        self.set_tag("Filterbasement", str(value))

    def get_tag(self, tag: str) -> None | ET.Element:
        """
        Recover the tag element.

        :param tag: The name of the tag.

        :return: The xml element.
        """
        element = self.xml.find(tag)
        return element  # type: ignore

    def set_tag(self, tag: str, value: str):
        """
        Set the value for the tag.

        :param tag: The name of the tag.
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
