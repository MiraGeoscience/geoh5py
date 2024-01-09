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

from ...curve import Curve
from ...object_base import ObjectType
from .base import AirborneEMSurvey, FEMSurvey

# pylint: disable=too-many-ancestors


class AirborneFEMSurvey(FEMSurvey, AirborneEMSurvey):
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

    @property
    def default_metadata(self) -> dict:
        """
        Default dictionary of metadata for AirborneFEM entities.
        """
        return {
            "EM Dataset": {
                "Channels": [],
                "Input type": "Rx",
                "Property groups": [],
                "Receivers": None,
                "Survey type": "Airborne FEM",
                "Transmitters": None,
                "Unit": "Hertz (Hz)",
            }
        }

    @property
    def default_receiver_type(self):
        """
        :return: Transmitter class
        """
        return AirborneFEMReceivers

    @property
    def default_transmitter_type(self):
        """
        :return: Transmitter class
        """
        return AirborneFEMTransmitters


class AirborneFEMReceivers(AirborneFEMSurvey):  # pylint: disable=too-many-ancestors
    """
    Airborne frequency-domain electromagnetic receivers class.
    """

    __TYPE_UID = uuid.UUID("{b3a47539-0301-4b27-922e-1dde9d882c60}")
    __TYPE = "Receivers"

    def __init__(self, object_type: ObjectType, name="Airborne FEM Rx", **kwargs):
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


class AirborneFEMTransmitters(AirborneFEMSurvey):  # pylint: disable=too-many-ancestors
    """
    Airborne frequency-domain electromagnetic transmitters class.
    """

    __TYPE_UID = uuid.UUID("{a006cf3e-e24a-4c02-b904-2e57b9b5916d}")
    __TYPE = "Transmitters"

    def __init__(self, object_type: ObjectType, name="Airborne FEM Tx", **kwargs):
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
