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

# pylint: disable=too-many-locals, too-many-branches

from __future__ import annotations

import uuid

from geoh5py.objects.object_base import ObjectType

from .base import LargeLoopGroundEMSurvey, MovingLoopGroundEMSurvey, TEMSurvey

# pylint: disable=too-many-ancestors, no-member
# mypy: disable-error-code="attr-defined"


class MovingLoopGroundTEMSurvey(TEMSurvey, MovingLoopGroundEMSurvey):
    @property
    def default_metadata(self) -> dict:
        """
        Default dictionary of metadata for AirborneTEM entities.
        """
        return {
            "EM Dataset": {
                "Channels": [],
                "Input type": "Tx and Rx",
                "Loop radius": 1,
                "Property groups": [],
                "Receivers": None,
                "Survey type": "Ground TEM",
                "Transmitters": None,
                "Unit": "Milliseconds (ms)",
                "Waveform": {"Timing mark": 0.0},
            }
        }

    @property
    def default_receiver_type(self):
        """
        :return: Transmitter class
        """
        return MovingLoopGroundTEMReceivers

    @property
    def default_transmitter_type(self):
        """
        :return: Transmitter class
        """
        return MovingLoopGroundTEMTransmitters


class MovingLoopGroundTEMReceivers(MovingLoopGroundTEMSurvey):  # pylint: disable=too-many-ancestors
    """
    Airborne time-domain electromagnetic receivers class.
    """

    __TYPE_UID = uuid.UUID("{41018a45-01a0-4c61-a7cb-9f32d8159df4}")
    __TYPE = "Receivers"

    _transmitters: MovingLoopGroundTEMTransmitters | None = None

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
    def default_transmitter_type(self):
        """
        :return: Transmitter class
        """
        return MovingLoopGroundTEMTransmitters

    @property
    def type(self):
        """Survey element type"""
        return self.__TYPE


class MovingLoopGroundTEMTransmitters(MovingLoopGroundTEMSurvey):  # pylint: disable=too-many-ancestors
    """
    Airborne frequency-domain electromagnetic transmitters class.
    """

    __TYPE_UID = uuid.UUID("{98a96d44-6144-4adb-afbe-0d5e757c9dfc}")
    __TYPE = "Transmitters"

    def __init__(self, object_type: ObjectType, name="Ground TEM Tx", **kwargs):
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
    def default_receiver_type(self):
        """
        :return: Transmitter class
        """
        return MovingLoopGroundTEMReceivers

    @property
    def type(self):
        """Survey element type"""
        return self.__TYPE


class LargeLoopGroundTEMSurvey(TEMSurvey, LargeLoopGroundEMSurvey):
    @property
    def default_metadata(self) -> dict:
        """
        Default dictionary of metadata for AirborneTEM entities.
        """
        return {
            "EM Dataset": {
                "Channels": [],
                "Input type": "Tx and Rx",
                "Property groups": [],
                "Receivers": None,
                "Survey type": "Ground TEM (large-loop)",
                "Transmitters": None,
                "Unit": "Milliseconds (ms)",
                "Waveform": {"Timing mark": 0.0},
            }
        }

    @property
    def default_receiver_type(self):
        """
        :return: Transmitter class
        """
        return LargeLoopGroundTEMReceivers

    @property
    def default_transmitter_type(self):
        """
        :return: Transmitter class
        """
        return LargeLoopGroundTEMTransmitters


class LargeLoopGroundTEMReceivers(LargeLoopGroundTEMSurvey):
    """
    Ground time-domain electromagnetic receivers class.
    """

    __TYPE_UID = uuid.UUID("{deebe11a-b57b-4a03-99d6-8f27b25eb2a8}")
    __TYPE = "Receivers"

    _transmitters: LargeLoopGroundTEMTransmitters | None = None

    def __init__(self, object_type: ObjectType, name="Ground TEM Rx", **kwargs):
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
    def default_transmitter_type(self):
        """
        :return: Transmitter class
        """
        return LargeLoopGroundTEMTransmitters

    @property
    def type(self):
        """Survey element type"""
        return self.__TYPE


class LargeLoopGroundTEMTransmitters(LargeLoopGroundTEMSurvey):
    """
    Ground time-domain electromagnetic transmitters class.
    """

    __TYPE_UID = uuid.UUID("{17dbbfbb-3ee4-461c-9f1d-1755144aac90}")
    __TYPE = "Transmitters"

    _receivers: LargeLoopGroundTEMReceivers | None = None

    def __init__(self, object_type: ObjectType, name="Ground TEM Tx", **kwargs):
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
    def default_receiver_type(self):
        """
        :return: Transmitter class
        """
        return LargeLoopGroundTEMReceivers

    @property
    def type(self):
        """Survey element type"""
        return self.__TYPE
