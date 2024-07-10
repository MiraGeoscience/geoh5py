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

import uuid

from geoh5py.objects.points import Points

from .base import FEMSurvey


class MTReceivers(FEMSurvey, Points):
    """
    A magnetotellurics survey object.
    """

    _TYPE_UID = uuid.UUID("{b99bd6e5-4fe1-45a5-bd2f-75fc31f91b38}")
    __TYPE = "Receivers"
    __INPUT_TYPE = ["Rx only"]

    def __init__(self, name="Magnetotellurics rx", **kwargs):
        super().__init__(name=name, **kwargs)

    @property
    def default_input_types(self) -> list[str]:
        """Choice of survey creation types."""
        return self.__INPUT_TYPE

    @property
    def default_metadata(self) -> dict:
        """
        :return: Default unique identifier
        """
        return {
            "EM Dataset": {
                "Channels": [],
                "Input type": "Rx only",
                "Property groups": [],
                "Receivers": None,
                "Survey type": "Magnetotellurics",
                "Unit": "Hertz (Hz)",
            }
        }

    @property
    def default_receiver_type(self):
        """
        :return: Transmitter class
        """
        return MTReceivers

    @property
    def default_transmitter_type(self):
        """
        :return: Transmitter class
        """
        return type(None)

    @property
    def base_receiver_type(self):
        return Points

    @property
    def base_transmitter_type(self):
        return type(None)

    @property
    def type(self):
        """Survey element type"""
        return self.__TYPE
