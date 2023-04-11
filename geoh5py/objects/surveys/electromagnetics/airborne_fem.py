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

import uuid

import numpy as np

from geoh5py.objects.curve import Curve
from geoh5py.objects.object_base import ObjectType

from .base import FEMSurvey, AirborneEMSurvey


class AirborneFEMSurvey(FEMSurvey, AirborneEMSurvey):  # pylint: disable=too-many-ancestors


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

    __TYPE_UID = uuid.UUID("{19730589-fd28-4649-9de0-ad47249d9aba}")
    __TYPE = "Receivers"

    def __init__(self, object_type: ObjectType, name="Airborne FEM Rx", **kwargs):
        super().__init__(object_type, name=name, **kwargs)

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

    __TYPE_UID = uuid.UUID("{58c4849f-41e2-4e09-b69b-01cf4286cded}")
    __TYPE = "Transmitters"

    def __init__(self, object_type: ObjectType, name="Airborne FEM Tx", **kwargs):
        super().__init__(object_type, name=name, **kwargs)

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
