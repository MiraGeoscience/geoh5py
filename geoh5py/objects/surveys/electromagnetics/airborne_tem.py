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

from geoh5py.objects.curve import Curve
from geoh5py.objects.object_base import ObjectType

from .base import AirborneEMSurvey, TEMSurvey


# pylint: disable=too-many-ancestors


class AirborneTEMSurvey(TEMSurvey, AirborneEMSurvey):
    @property
    def default_metadata(self) -> dict:
        """
        Default dictionary of metadata for AirborneTEM entities.
        """
        return {
            "EM Dataset": {
                "Channels": [],
                "Input type": "Rx",
                "Property groups": [],
                "Receivers": None,
                "Survey type": "Airborne TEM",
                "Transmitters": None,
                "Unit": "Milliseconds (ms)",
                "Waveform": {"Timing mark": 0.0},
            }
        }

    @property
    def default_receiver_type(self):
        """
        :return: Receiver class
        """
        return AirborneTEMReceivers

    @property
    def default_transmitter_type(self):
        """
        :return: Transmitter class
        """
        return AirborneTEMTransmitters

    @property
    def base_receiver_type(self):
        """
        :return: Base receiver class
        """
        return Curve

    @property
    def base_transmitter_type(self):
        """
        :return: Base transmitter class
        """
        return Curve


class AirborneTEMReceivers(AirborneTEMSurvey):
    """
    Airborne time-domain electromagnetic receivers class.
    """

    __TYPE_UID = uuid.UUID("{19730589-fd28-4649-9de0-ad47249d9aba}")
    __TYPE = "Receivers"

    def __init__(self, object_type: ObjectType, name="Airborne TEM Rx", **kwargs):
        super().__init__(object_type, name=name, **kwargs)

    @property
    def complement(self):
        return self.transmitters

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        """
        :return: Default unique identifier
        """
        return cls.__TYPE_UID

    @property
    def type(self):
        """Survey element type"""
        return self.__TYPE


class AirborneTEMTransmitters(AirborneTEMSurvey):
    """
    Airborne time-domain electromagnetic transmitters class.
    """

    __TYPE_UID = uuid.UUID("{58c4849f-41e2-4e09-b69b-01cf4286cded}")
    __TYPE = "Transmitters"

    def __init__(self, object_type: ObjectType, name="Airborne TEM Tx", **kwargs):
        super().__init__(object_type, name=name, **kwargs)

    @property
    def complement(self):
        return self.receivers

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        """
        :return: Default unique identifier
        """
        return cls.__TYPE_UID

    @property
    def type(self):
        """Survey element type"""
        return self.__TYPE
