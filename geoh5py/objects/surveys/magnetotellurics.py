#  Copyright (c) 2021 Mira Geoscience Ltd.
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

import numpy as np

from ..curve import Points
from ..object_type import ObjectType


class Magnetotellurics(Points):
    """
    A magnetotelluric survey object.
    """

    __TYPE_UID = uuid.UUID("{b99bd6e5-4fe1-45a5-bd2f-75fc31f91b38}")
    _input_type = "Rx Only"
    _survey_type = "Magnetotellurics"
    _unit = "Hertz (Hz)"
    _default_metadata = {
        "EM Dataset": {
            "Channels": [],
            "Input type": _input_type,
            "Property groups": [],
            "Receivers": "",
            "Survey type": _survey_type,
            "Unit": _unit,
        }
    }

    def __init__(self, object_type: ObjectType, **kwargs):
        super().__init__(object_type, **kwargs)

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        """
        :return: Default unique identifier
        """
        return cls.__TYPE_UID

    @property
    def input_type(self):
        """Type of measurements"""
        return self._input_type

    @property
    def survey_type(self):
        """Type of EM survey"""
        return self._survey_type

    @property
    def unit(self):
        """Data unit"""
        return self._unit

    @property
    def metadata(self) -> dict | None:
        """
        Metadata attached to the entity.
        """
        if getattr(self, "_metadata", None) is None:
            metadata = self.workspace.fetch_metadata(self.uid)

            if metadata is None:
                self._default_metadata["EM Dataset"]["Receivers"] = str(self.uid)
                metadata = self._default_metadata

            self._metadata = metadata
        return self._metadata

    @metadata.setter
    def metadata(self, values: dict):

        if "EM Dataset" not in values.keys():
            raise KeyError("'EM Dataset' must be a 'metadata' key")

        for key in self._default_metadata:
            if key not in values:
                raise KeyError(f"{key} argument missing from the input metadata.")

        self._metadata = values
        self.modified_attributes = "metadata"

    @property
    def channels(self):
        """
        List of measured frequencies.
        """
        if self.metadata is None:
            raise AttributeError("No metadata set.")

        channels = self.metadata["EM Dataset"]["Channels"]
        return channels

    @channels.setter
    def channels(self, values: np.ndarray):
        if not isinstance(values, np.ndarray):
            raise ValueError(f"The list of channels must of type {np.ndarray}")

        if self.metadata is None:
            raise AttributeError("No metadata set.")

        self.metadata["EM Dataset"]["Channels"] = values
        self.modified_attributes = "metadata"
