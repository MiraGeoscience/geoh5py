#  Copyright (c) 2022 Mira Geoscience Ltd.
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

import uuid

from geoh5py.objects.object_type import ObjectType

from .base import BaseEMSurvey


class Magnetotellurics(BaseEMSurvey):
    """
    A magnetotellurics survey object.
    """

    __TYPE_UID = uuid.UUID("{b99bd6e5-4fe1-45a5-bd2f-75fc31f91b38}")
    __METADATA = {
        "EM Dataset": {
            "Channels": [],
            "Input type": "Rx only",
            "Property groups": [],
            "Receivers": "",
            "Survey type": "Magnetotellurics",
            "Unit": "Hertz (Hz)",
        }
    }
    __UNITS = [
        "Hertz (Hz)",
        "KiloHertz (kHz)",
        "MegaHertz (MHz)",
        "Gigahertz (GHz)",
    ]
    __INPUT_TYPE = ["Rx only"]

    def __init__(self, object_type: ObjectType, **kwargs):
        super().__init__(object_type, **kwargs)

    @property
    def default_input_types(self) -> list[str]:
        """Input types. Must be 'Rx only'"""
        return self.__INPUT_TYPE

    @property
    def default_metadata(self) -> dict:
        """
        :return: Default unique identifier
        """
        return self.__METADATA.copy()

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        """
        :return: Default unique identifier
        """
        return cls.__TYPE_UID

    @property
    def default_units(self) -> list[str]:
        """Accepted time units. Must be one of "Seconds (s)",
        "Milliseconds (ms)", "Microseconds (us)" or "Nanoseconds (ns)"
        """
        return self.__UNITS

    @property
    def metadata(self) -> dict:
        """
        Metadata attached to the entity.
        """
        if getattr(self, "_metadata", None) is None:
            metadata = self.workspace.fetch_metadata(self.uid)

            if metadata is None:
                self.default_metadata["EM Dataset"]["Receivers"] = self.uid
                metadata = self.default_metadata

            self._metadata = metadata
        return self._metadata

    @metadata.setter
    def metadata(self, values: dict):

        if not isinstance(values, dict):
            raise TypeError("'metadata' must be of type 'dict'")

        if "EM Dataset" not in values:
            raise KeyError("'EM Dataset' must be a 'metadata' key")

        for key in self.default_metadata["EM Dataset"]:
            if key not in values["EM Dataset"]:
                raise KeyError(f"'{key}' argument missing from the input metadata.")

        self._metadata = values
        self.modified_attributes = "metadata"

    @property
    def receivers(self):
        """MT receivers"""
        return self
