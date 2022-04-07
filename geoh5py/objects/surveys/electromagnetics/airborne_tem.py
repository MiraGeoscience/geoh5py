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

from __future__ import annotations

import uuid

import numpy as np

from geoh5py.objects import Curve
from geoh5py.objects.object_type import ObjectType

from .base import BaseEMSurvey


class BaseAirborneTEM(BaseEMSurvey, Curve):
    __MAP = {
        "crossline_offset": "Crossline offset",
        "inline_offset": "Inline offset",
        "pitch": "Pitch",
        "roll": "Roll",
        "vertical_offset": "Vertical offset",
        "yaw": "Yaw",
    }
    __UNITS = [
        "Seconds (s)",
        "Milliseconds (ms)",
        "Microseconds (us)",
        "Nanoseconds (ns)",
    ]
    __INPUT_TYPE = ["Rx", "Tx", "Tx and Rx"]

    def __init__(
        self,
        object_type: ObjectType,
        **kwargs,
    ):

        super().__init__(
            object_type,
            **kwargs,
        )

    @property
    def default_input_types(self) -> list[str]:
        """Input types. Must be one of 'Rx', 'Tx', 'Tx and Rx'."""
        return self.__INPUT_TYPE

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
            }
        }

    @property
    def default_units(self) -> list[str]:
        """Accepted time units. Must be one of "Seconds (s)",
        "Milliseconds (ms)", "Microseconds (us)" or "Nanoseconds (ns)"
        """
        return self.__UNITS

    def fetch_metadata(self, key: str) -> float | uuid.UUID | None:
        """
        Fetch entry from the metadata.
        """
        field = self.__MAP[key]
        if field + " value" in self.metadata["EM Dataset"]:
            return self.metadata["EM Dataset"][field + " value"]
        if field + " property" in self.metadata["EM Dataset"]:
            return self.metadata["EM Dataset"][field + " property"]
        return None

    @property
    def crossline_offset(self) -> float | uuid.UUID | None:
        """
        Numeric value or property UUID for the crossline offset between receiver and transmitter.
        """
        return self.fetch_metadata("crossline_offset")

    @crossline_offset.setter
    def crossline_offset(self, value: float | uuid.UUID | None):
        self.set_metadata("crossline_offset", value)

    @property
    def inline_offset(self) -> float | uuid.UUID | None:
        """
        Numeric value or property UUID for the inline offset between receiver and transmitter.
        """
        return self.fetch_metadata("inline_offset")

    @inline_offset.setter
    def inline_offset(self, value: float | uuid.UUID):
        self.set_metadata("inline_offset", value)

    @property
    def loop_radius(self) -> float | None:
        """Transmitter loop radius"""
        return self.metadata["EM Dataset"].get("Loop radius", None)

    @loop_radius.setter
    def loop_radius(self, value: float | None):
        if not isinstance(value, (float, type(None))):
            raise TypeError("Input 'loop_radius' must be of type 'float'")
        self.edit_metadata({"Loop radius": value})

    @property
    def pitch(self) -> float | uuid.UUID | None:
        """
        Numeric value or property UUID for the pitch angle of the transmitter loop.
        """
        return self.fetch_metadata("pitch")

    @pitch.setter
    def pitch(self, value: float | uuid.UUID | None):
        self.set_metadata("pitch", value)

    @property
    def relative_to_bearing(self) -> bool | None:
        """Data relative_to_bearing"""
        return self.metadata["EM Dataset"].get("Angles relative to bearing", None)

    @relative_to_bearing.setter
    def relative_to_bearing(self, value: bool | None):
        if not isinstance(value, (bool, type(None))):
            raise TypeError("Input 'relative_to_bearing' must be one of type 'bool'")
        self.edit_metadata({"Angles relative to bearing": value})

    @property
    def roll(self) -> float | uuid.UUID | None:
        """
        Numeric value or property UUID for the roll angle of the transmitter loop.
        """
        return self.fetch_metadata("roll")

    @roll.setter
    def roll(self, value: float | uuid.UUID | None):
        self.set_metadata("roll", value)

    def set_metadata(self, key: str, value: float | uuid.UUID | None):
        field = self.__MAP[key]
        if isinstance(value, float):
            self.edit_metadata({field + " value": value, field + " property": None})
        elif isinstance(value, uuid.UUID):
            self.edit_metadata({field + " value": None, field + " property": value})
        elif value is None:
            self.edit_metadata({field + " value": None, field + " property": None})
        else:
            raise TypeError(
                f"Input '{key}' must be one of type float, uuid.UUID or None"
            )

    @property
    def timing_mark(self) -> float | None:
        """
        Timing mark from the beginning of the discrete :attr:`waveform`.
        Generally used as the reference (time=0.0) for the provided
        (-) on-time an (+) off-time :attr:`channels`.
        """
        if (
            "Waveform" in self.metadata["EM Dataset"]
            and "Timing mark" in self.metadata["EM Dataset"]["Waveform"]
        ):
            timing_mark = self.metadata["EM Dataset"]["Waveform"]["Timing mark"]
            return timing_mark

        return None

    @timing_mark.setter
    def timing_mark(self, timing_mark: float | None):
        if not isinstance(timing_mark, (float, type(None))):
            raise ValueError("Input timing_mark must be a float or None.")

        if self.waveform is not None:
            value = self.metadata["EM Dataset"]["Waveform"]
        else:
            value = {}

        if timing_mark is None and "Timing mark" in value:
            del value["Timing mark"]
        else:
            value["Timing mark"] = timing_mark

        self.edit_metadata({"Waveform": value})

    @property
    def vertical_offset(self) -> float | uuid.UUID | None:
        """
        Numeric value or property UUID for the vertical offset between receiver and transmitter.
        """
        return self.fetch_metadata("vertical_offset")

    @vertical_offset.setter
    def vertical_offset(self, value: float | uuid.UUID | None):
        self.set_metadata("vertical_offset", value)

    @property
    def waveform(self) -> np.ndarray | None:
        """
        Discrete waveform of the TEM source provided as
        :obj:`numpy.array` of type :obj:`float`, shape(n, 2)

        .. code-block:: python

            waveform = [
                [time_1, current_1],
                [time_2, current_2],
                ...
            ]

        """
        if (
            "Waveform" in self.metadata["EM Dataset"]
            and "Discretization" in self.metadata["EM Dataset"]["Waveform"]
        ):
            waveform = np.vstack(
                [
                    [row["time"], row["current"]]
                    for row in self.metadata["EM Dataset"]["Waveform"]["Discretization"]
                ]
            )
            return waveform
        return None

    @waveform.setter
    def waveform(self, waveform: np.ndarray | None):
        if not isinstance(waveform, (np.ndarray, type(None))):
            raise TypeError("Input waveform must be a numpy.ndarray or None.")

        if isinstance(waveform, np.ndarray):
            if waveform.ndim != 2 or waveform.shape[1] != 2:
                raise ValueError(
                    "Input waveform must be a numpy.ndarray of shape (*, 2)."
                )

            if self.timing_mark is not None:
                value = self.metadata["EM Dataset"]["Waveform"]
            else:
                value = {}

            value["Discretization"] = [
                {"current": row[1], "time": row[0]} for row in waveform
            ]

        else:
            value = waveform

        self.edit_metadata({"Waveform": value})

    @property
    def yaw(self) -> float | uuid.UUID | None:
        """
        Numeric value or property UUID for the yaw angle of the transmitter loop.
        """
        return self.fetch_metadata("yaw")

    @yaw.setter
    def yaw(self, value: float | uuid.UUID):
        self.set_metadata("yaw", value)


class AirborneTEMReceivers(BaseAirborneTEM):
    """
    Airborne time-domain electromagnetic receivers class.
    """

    __TYPE_UID = uuid.UUID("{19730589-fd28-4649-9de0-ad47249d9aba}")
    __TYPE = "Receivers"

    def __init__(self, object_type: ObjectType, **kwargs):
        super().__init__(object_type, **kwargs)

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
        return AirborneTEMTransmitters

    @property
    def type(self):
        """Survey element type"""
        return self.__TYPE


class AirborneTEMTransmitters(BaseAirborneTEM):
    """
    Airborne time-domain electromagnetic transmitters class.
    """

    __TYPE_UID = uuid.UUID("{58c4849f-41e2-4e09-b69b-01cf4286cded}")
    __TYPE = "Transmitters"

    def __init__(self, object_type: ObjectType, **kwargs):
        super().__init__(object_type, **kwargs)

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
        return AirborneTEMReceivers

    @property
    def type(self):
        """Survey element type"""
        return self.__TYPE
