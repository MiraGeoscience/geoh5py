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

from .base import FEMSurvey, LargeLoopGroundEMSurvey, MovingLoopGroundEMSurvey


# pylint: disable=too-many-ancestors


class MovingLoopGroundFEMSurvey(FEMSurvey, MovingLoopGroundEMSurvey):
    @property
    def default_input_types(self) -> list[str]:
        """Choice of survey creation types."""
        return self.__INPUT_TYPE

    @property
    def default_metadata(self) -> dict:
        """
        Default dictionary of metadata for MovingLoopGroundFEMSurvey entities.
        """
        return {
            "EM Dataset": {
                "Channels": [],
                "Input type": "Tx and Rx",
                "Loop radius": 1,
                "Property groups": [],
                "Receivers": None,
                "Survey type": "Ground FEM",
                "Transmitters": None,
                "Unit": "Hertz (Hz)",
            }
        }

    @property
    def default_receiver_type(self):
        """
        :return: Transmitter class
        """
        return MovingLoopGroundFEMReceivers

    @property
    def default_transmitter_type(self):
        """
        :return: Transmitter class
        """
        return MovingLoopGroundFEMTransmitters


class MovingLoopGroundFEMReceivers(MovingLoopGroundFEMSurvey):  # pylint: disable=too-many-ancestors
    """
    Airborne frequency-domain electromagnetic receivers class.
    """

    _TYPE_UID = uuid.UUID("{a81c6b0a-f290-4bc8-b72d-60e59964bfe8}")
    __TYPE = "Receivers"

    _transmitters: MovingLoopGroundFEMTransmitters | None = None
    _default_name = "Airborne FEM Rx"

    @property
    def complement(self):
        return self.transmitters

    @property
    def default_transmitter_type(self):
        """
        :return: Transmitter class
        """
        return MovingLoopGroundFEMTransmitters

    @property
    def type(self):
        """Survey element type"""
        return self.__TYPE


class MovingLoopGroundFEMTransmitters(MovingLoopGroundFEMSurvey):  # pylint: disable=too-many-ancestors
    """
    Airborne frequency-domain electromagnetic transmitters class.
    """

    _TYPE_UID = uuid.UUID("{f59d5a1c-5e63-4297-b5bc-43898cb4f5f8}")
    __TYPE = "Transmitters"
    _default_name = "Ground FEM Tx"

    @property
    def complement(self):
        return self.receivers

    @property
    def default_receiver_type(self):
        """
        :return: Transmitter class
        """
        return MovingLoopGroundFEMReceivers

    @property
    def type(self):
        """Survey element type"""
        return self.__TYPE


class LargeLoopGroundFEMSurvey(FEMSurvey, LargeLoopGroundEMSurvey):
    @property
    def default_metadata(self) -> dict:
        """
        Default dictionary of metadata for AirborneFEM entities.
        """
        return {
            "EM Dataset": {
                "Channels": [],
                "Input type": "Tx and Rx",
                "Property groups": [],
                "Receivers": None,
                "Survey type": "Ground FEM (large-loop)",
                "Transmitters": None,
                "Unit": "Hertz (Hz)",
            }
        }

    @property
    def default_receiver_type(self):
        """
        :return: Transmitter class
        """
        return LargeLoopGroundFEMReceivers

    @property
    def default_transmitter_type(self):
        """
        :return: Transmitter class
        """
        return LargeLoopGroundFEMTransmitters


class LargeLoopGroundFEMReceivers(LargeLoopGroundFEMSurvey):  # pylint: disable=too-many-ancestors
    """
    Airborne frequency-domain electromagnetic receivers class.
    """

    _TYPE_UID = uuid.UUID("{30928322-cf2c-4230-b393-4dc629259b64}")
    __TYPE = "Receivers"

    _transmitters: LargeLoopGroundFEMTransmitters | None = None
    _default_name = "Airborne FEM Rx"

    @property
    def complement(self):
        return self.transmitters

    @property
    def default_transmitter_type(self):
        """
        :return: Transmitter class
        """
        return LargeLoopGroundFEMTransmitters

    @property
    def type(self):
        """Survey element type"""
        return self.__TYPE


class LargeLoopGroundFEMTransmitters(LargeLoopGroundFEMSurvey):  # pylint: disable=too-many-ancestors
    """
    Airborne frequency-domain electromagnetic transmitters class.
    """

    _TYPE_UID = uuid.UUID("{fe1a240a-9189-49ff-aa7e-6067405b6e0a}")
    __TYPE = "Transmitters"
    _default_name = "Ground FEM Tx"

    @property
    def complement(self):
        return self.receivers

    @property
    def default_receiver_type(self):
        """
        :return: Transmitter class
        """
        return LargeLoopGroundFEMReceivers

    @property
    def type(self):
        """Survey element type"""
        return self.__TYPE
