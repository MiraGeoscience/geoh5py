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

from geoh5py.objects.object_type import ObjectType

from .base import BaseEMSurvey


class BaseAirborneTEM(BaseEMSurvey):
    __METADATA = {
        "EM Dataset": {
            "Channels": [],
            "Input type": "Rx",
            "Property groups": [],
            "Receivers": None,
            "Survey type": "Airborne TEM",
            "Transmitters": None,
            "Unit": "Milliseconds (ms)",
            "Yaw value": 0,
            "Pitch value": 0,
            "Roll value": 0,
            "Inline offset value": 0,
            "Crossline offset value": 0,
            "Vertical offset value": 0,
            "Loop radius": 1,
        }
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
        return self.__METADATA

    @property
    def default_units(self) -> list[str]:
        """Accepted time units. Must be one of "Seconds (s)",
        "Milliseconds (ms)", "Microseconds (us)" or "Nanoseconds (ns)"
        """
        return self.__UNITS

    @property
    def crossline_offset(self) -> float | uuid.UUID | None:
        """
        Numeric value or property UUID for the crossline offset between receiver and transmitter.
        """
        if "Crossline offset value" in self.metadata["EM Dataset"]:
            return self.metadata["EM Dataset"]["Crossline offset value"]
        if "Crossline offset property" in self.metadata["EM Dataset"]:
            return self.metadata["EM Dataset"]["Crossline offset property"]
        return None

    @crossline_offset.setter
    def crossline_offset(self, value: float | uuid.UUID):
        if isinstance(value, float):
            self.edit_metadata(
                {"Crossline offset value": value, "Crossline offset property": None}
            )
        elif isinstance(value, uuid.UUID):
            self.edit_metadata(
                {"Crossline offset value": None, "Crossline offset property": value}
            )
        else:
            raise TypeError(
                "Input 'crossline_offset' must be one of type float or uuid.UUID"
            )

    @property
    def inline_offset(self) -> float | uuid.UUID | None:
        """
        Numeric value or property UUID for the inline offset between receiver and transmitter.
        """
        if "Inline offset value" in self.metadata["EM Dataset"]:
            return self.metadata["EM Dataset"]["Inline offset value"]
        if "Inline offset property" in self.metadata["EM Dataset"]:
            return self.metadata["EM Dataset"]["Inline offset property"]
        return None

    @inline_offset.setter
    def inline_offset(self, value: float | uuid.UUID):
        if isinstance(value, float):
            self.edit_metadata(
                {"Inline offset value": value, "Inline offset property": None}
            )
        elif isinstance(value, uuid.UUID):
            self.edit_metadata(
                {"Inline offset value": None, "Inline offset property": value}
            )
        else:
            raise TypeError(
                "Input 'inline_offset' must be one of type float or uuid.UUID"
            )

    @property
    def loop_radius(self) -> float | None:
        """Transmitter loop radius"""
        return self.metadata["EM Dataset"].get("Loop radius", None)

    @loop_radius.setter
    def loop_radius(self, value: float):
        if not isinstance(value, float):
            raise TypeError("Input 'loop_radius' must be of type 'float'")
        self.edit_metadata({"Loop radius": value})

    @property
    def pitch(self) -> float | uuid.UUID | None:
        """
        Numeric value or property UUID for the pitch angle(s) of the transmitter coil.
        """
        if "Pitch value" in self.metadata["EM Dataset"]:
            return self.metadata["EM Dataset"]["Pitch value"]
        if "Pitch property" in self.metadata["EM Dataset"]:
            return self.metadata["EM Dataset"]["Pitch property"]
        return None

    @pitch.setter
    def pitch(self, value: float | uuid.UUID):
        if isinstance(value, float):
            self.edit_metadata({"Pitch value": value, "Pitch property": None})
        elif isinstance(value, uuid.UUID):
            self.edit_metadata({"Pitch value": None, "Pitch property": value})
        else:
            raise TypeError("Input 'pitch' must be one of type float or uuid.UUID")

    @property
    def relative_to_bearing(self) -> bool | None:
        """Data relative_to_bearing"""
        return self.metadata["EM Dataset"].get("Angles relative to bearing", None)

    @relative_to_bearing.setter
    def relative_to_bearing(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("Input 'relative_to_bearing' must be one of type 'bool'")
        self.edit_metadata({"Angles relative to bearing": value})

    @property
    def roll(self) -> float | uuid.UUID | None:
        """
        Numeric value or property UUID for the roll angle(s) of the transmitter coil.
        """
        if "Roll value" in self.metadata["EM Dataset"]:
            return self.metadata["EM Dataset"]["Roll value"]
        if "Roll property" in self.metadata["EM Dataset"]:
            return self.metadata["EM Dataset"]["Roll property"]
        return None

    @roll.setter
    def roll(self, value: float | uuid.UUID):
        if isinstance(value, float):
            self.edit_metadata({"Roll value": value, "Roll property": None})
        elif isinstance(value, uuid.UUID):
            self.edit_metadata({"Roll value": None, "Roll property": value})
        else:
            raise TypeError("Input 'roll' must be one of type float or uuid.UUID")

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
    def timing_mark(self, timing_mark: float):
        if not isinstance(timing_mark, float):
            raise ValueError("Input timing_mark must be a float.")

        self.edit_metadata({"Timing mark": timing_mark})

    @property
    def vertical_offset(self) -> float | uuid.UUID | None:
        """
        Numeric value or property UUID for the vertical offset between receiver and transmitter.
        """
        if "Vertical offset value" in self.metadata["EM Dataset"]:
            return self.metadata["EM Dataset"]["Vertical offset value"]
        if "Vertical offset property" in self.metadata["EM Dataset"]:
            return self.metadata["EM Dataset"]["Vertical offset property"]
        return None

    @vertical_offset.setter
    def vertical_offset(self, value: float | uuid.UUID):
        if isinstance(value, float):
            self.edit_metadata(
                {"Vertical offset value": value, "Vertical offset property": None}
            )
        elif isinstance(value, uuid.UUID):
            self.edit_metadata(
                {"Vertical offset value": None, "Vertical offset property": value}
            )
        else:
            raise TypeError(
                "Input 'vertical_offset' must be one of type float or uuid.UUID"
            )

    @property
    def waveform(self) -> np.ndarray | None:
        """
        :obj:`numpy.array` of :obj:`float`, shape(*, 2): Discrete waveform of the TEM source.

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

        if waveform is None and "Waveform" in self.metadata["EM Dataset"]:
            del self.metadata["EM Dataset"]["Waveform"]

        elif isinstance(waveform, np.ndarray):
            if waveform.ndim != 2 or waveform.shape[1] != 2:
                raise ValueError(
                    "Input waveform must be a numpy.ndarray of shape (*, 2)."
                )

            self.edit_metadata(
                {"Waveform": [{"current": row[1], "time": row[0]} for row in waveform]},
            )
        else:
            raise TypeError("Input waveform must be a numpy.ndarray or None.")

    @property
    def yaw(self) -> float | uuid.UUID | None:
        """
        Numeric value or property UUID for the yaw angle(s) of the transmitter coil.
        """
        if "Yaw value" in self.metadata["EM Dataset"]:
            return self.metadata["EM Dataset"]["Yaw value"]
        if "Yaw property" in self.metadata["EM Dataset"]:
            return self.metadata["EM Dataset"]["Yaw property"]
        return None

    @yaw.setter
    def yaw(self, value: float | uuid.UUID):
        if isinstance(value, float):
            self.edit_metadata({"Yaw value": value, "Yaw property": None})
        elif isinstance(value, uuid.UUID):
            self.edit_metadata({"Yaw value": None, "Yaw property": value})
        else:
            raise TypeError("Input 'yaw' must be one of type float or uuid.UUID")


class AirborneTEMReceivers(BaseAirborneTEM):
    """
    Airborne time-domain electromagnetic receivers class.
    """

    __TYPE_UID = uuid.UUID("{19730589-fd28-4649-9de0-ad47249d9aba}")

    def __init__(self, object_type: ObjectType, **kwargs):
        super().__init__(object_type, **kwargs)

    def copy(self, parent=None, copy_children: bool = True) -> AirborneTEMReceivers:
        """
        Function to copy a AirborneTEMReceivers to a different parent entity.

        :param parent: Target parent to copy the entity under. Copied to current
            :obj:`~geoh5py.shared.entity.Entity.parent` if None.
        :param copy_children: Create copies of AirborneTEMReceivers along with it.

        :return entity: Registered AirborneTEMReceivers to the workspace.
        """
        if parent is None:
            parent = self.parent

        omit_list = ["_metadata", "_receivers", "_transmitters"]
        new_entity = parent.workspace.copy_to_parent(
            self, parent, copy_children=copy_children, omit_list=omit_list
        )
        new_transmitters = parent.workspace.copy_to_parent(
            self.transmitters,
            parent,
            copy_children=copy_children,
            omit_list=omit_list,
        )
        new_entity.transmitters = new_transmitters
        parent.workspace.finalize()

        return new_entity

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
    def receivers(self) -> AirborneTEMReceivers:
        """
        The associated TEM receivers.
        """
        return self

    @receivers.setter
    def receivers(self, value: BaseEMSurvey):
        raise UserWarning(
            "Attribute 'receivers' of the class 'AirborneTEMReceivers' must reference to self. "
            f"Re-assignment to {value} ignored."
        )


class AirborneTEMTransmitters(BaseAirborneTEM):
    """
    Airborne time-domain electromagnetic transmitters class.
    """

    __TYPE_UID = uuid.UUID("{58c4849f-41e2-4e09-b69b-01cf4286cded}")

    def __init__(self, object_type: ObjectType, **kwargs):
        super().__init__(object_type, **kwargs)

    def copy(self, parent=None, copy_children: bool = True) -> AirborneTEMTransmitters:
        """
        Function to copy a AirborneTEMTransmitters to a different parent entity.

        :param parent: Target parent to copy the entity under. Copied to current
            :obj:`~geoh5py.shared.entity.Entity.parent` if None.
        :param copy_children: Create copies of AirborneTEMReceivers along with it.

        :return entity: Registered AirborneTEMTransmitters to the workspace.
        """
        if parent is None:
            parent = self.parent

        omit_list = ["_metadata", "_receivers", "_transmitters"]
        new_entity = parent.workspace.copy_to_parent(
            self, parent, copy_children=copy_children, omit_list=omit_list
        )
        new_receivers = parent.workspace.copy_to_parent(
            self.receivers,
            parent,
            copy_children=copy_children,
            omit_list=omit_list,
        )
        new_entity.receivers = new_receivers
        parent.workspace.finalize()

        return new_entity

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
    def transmitters(self) -> AirborneTEMTransmitters:
        """
        The associated current electrode object (sources).
        """
        return self

    @transmitters.setter
    def transmitters(self, value: BaseEMSurvey):
        raise UserWarning(
            "Attribute 'transmitters' of the class 'AirborneTEMTransmitters' "
            f"must reference to self. Re-assignment to {value} ignored."
        )
