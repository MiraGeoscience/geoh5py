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
from warnings import warn

from .text_data import TextData

PARAMETERS = ["Colour"]
ATTRIBUTES = ["tag", "text", "attrib"]


class VisualParameters(TextData):
    _xml: ET.Element | None = None

    @property
    def xml(self) -> ET.Element | None:
        """
        :obj:`str` XML string.
        """
        if self._xml is None and isinstance(self.values, str):
            self.xml = self.values  # type: ignore

        return self._xml

    @xml.setter
    def xml(self, values: str):
        if not isinstance(values, str):
            raise TypeError(f"Input 'values' for {self} must be of type {str}.")

        try:
            self._xml = ET.fromstring(values)
        except ET.ParseError as exception:
            raise ValueError(
                f"Input 'values' for {self} must be a valid XML string ({exception})."
            ) from exception

    @property
    def colour(self) -> None | str | dict:
        """
        :obj:`dict` the Colour dictionary.
        """
        return self.get_child("Colour", "text")

    @colour.setter
    def colour(self, values: str):
        # todo: check if values has the right format: what is this format???
        if not isinstance(values, str):
            raise TypeError(f"Input 'values' for {self} must be of type {str}.")

        self.modify_xml({"tag": "Colour", "attrib": {}, "text": values})

    def modify_xml(self, values: dict):
        """
        Modify the XML Element with a dictionary.
        :param values: a dictionary of the values to modify
        that must contains 'tag', 'attrib' and 'text' keys.
        """
        if not isinstance(values, dict):
            raise TypeError(f"Input 'values' for {self} must be of type {dict}.")

        self.set_child(values["tag"], "text", values["text"])
        self.set_child(values["tag"], "attrib", values["attrib"])

    def check_child(self, child: str, attribute: str) -> bool:
        """
        Check if the child is a valid parameter.
        :param child: the key of the child to check.
        :param attribute: the attribute to check.
        :return: True if the child and attribute are valid.
        """
        if child not in PARAMETERS:
            warn(f"{child} is not a valid parameter.")
            return False

        if not isinstance(self.xml, ET.Element):
            warn(f"XML is not set for {self}.")
            return False

        if self.xml.find(child) is None:
            warn(f"{child} is not present in the XML.")
            return False

        if not all(
            hasattr(self.xml.find(child), attribute_) for attribute_ in ATTRIBUTES
        ):
            warn(f"{child} does not have the required attributes ({ATTRIBUTES}).")
            return False

        if attribute not in ATTRIBUTES:
            warn(f"Input 'attribute' for {self} must be in {ATTRIBUTES}.")
            return False

        return True

    def get_child(self, child: str, attribute: str) -> None | str | dict:
        """
        Check if the child is a valid parameter.
        :param child: the key of the child to check.
        :param attribute: the attribute to check.
        :return: True if the child is a valid parameter.
        """
        if self.check_child(child, attribute):
            return getattr(self.xml.find(child), attribute)  # type: ignore
        return None

    def set_child(self, child: str, attribute: str, value: str | dict):
        """
        Check if the child is a valid parameter.
        :param child: the key of the child to check.
        :param attribute: the attribute to check.
        :param value: the value to set.
        """
        if self.check_child(child, attribute):
            setattr(self.xml.find(child), attribute, value)  # type: ignore
