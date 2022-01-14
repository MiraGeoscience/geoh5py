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
from typing import Any

import numpy as np

from ..curve import Curve
from ..object_type import ObjectType


class BaseAirborneTEM(Curve):
    __META_KEY = "EM Dataset"
    __default_metadata = {
        __META_KEY: {
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
            "Loop radius": 1,
        }
    }

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
    def channels(self):
        """
        List of measured time channels.
        """
        channels = self.metadata["EM Dataset"]["Channels"]
        return channels

    @channels.setter
    def channels(self, values: list | np.ndarray):
        if isinstance(values, np.ndarray):
            values = values.tolist()

        if not isinstance(values, list) or not np.all(
            [isinstance(x, float) for x in values]
        ):
            raise TypeError(
                f"Values provided as 'channels' must be a list of {float}. {type(values)} provided"
            )

        self.edit_metadata({"Channels": values})

    def copy(self, parent=None, copy_children: bool = True):
        """
        Function to copy a survey to a different parent entity.

        :param parent: Target parent to copy the entity under. Copied to current
            :obj:`~geoh5py.shared.entity.Entity.parent` if None.
        :param copy_children: Create copies of all children entities along with it.

        :return entity: Registered Entity to the workspace.
        """
        if parent is None:
            parent = self.parent

        omit_list = ["_metadata", "_receivers", "_transmitters"]
        new_entity = parent.workspace.copy_to_parent(
            self, parent, copy_children=copy_children, omit_list=omit_list
        )

        if "Receivers" in type(self).__name__:
            new_transmitters = parent.workspace.copy_to_parent(
                self.transmitters,
                parent,
                copy_children=copy_children,
                omit_list=omit_list,
            )
            new_entity.transmitters = new_transmitters
        else:
            new_transmitters = parent.workspace.copy_to_parent(
                self.transmitters,
                parent,
                copy_children=copy_children,
                omit_list=omit_list,
            )
            new_entity.transmitters = new_transmitters

        parent.workspace.finalize()

        return new_entity

    @property
    def default_metadata(self) -> dict:
        """
        :return: Default dictionary of metadata for AirborneTEM entities.
        """
        return self.__default_metadata.copy()

    def edit_metadata(self, entries: dict[str, Any]):
        """
        Utility function to edit or add metadata fields and trigger an update
        on the receiver and transmitter entities.

        :param entries: Metadata key: value pairs.
        """
        for key, value in entries.items():
            if key in ["Discretization", "Timing mark", "Waveform"]:
                if "Waveform" not in self.metadata["EM Dataset"]:
                    self.metadata["EM Dataset"]["Waveform"] = {}

                wave_key = key.replace("Waveform", "Discretization")
                self.metadata["EM Dataset"]["Waveform"][wave_key] = value

            elif key == "Property groups":
                if not isinstance(value, list):
                    value = [value]

                for val in value:
                    if not isinstance(
                        val, str
                    ) or not self.find_or_create_property_group(name=val):
                        raise ValueError(f"No property_group with name '{val}' found.")

                    prop_group = self.find_or_create_property_group(name=val)

                    if len(prop_group.properties) != len(self.channels):
                        raise ValueError(
                            f"Number of properties in group '{prop_group.name}' "
                            + "differ from the number of 'channels'."
                        )

                self.metadata["EM Dataset"][key] = value

            elif value is None:
                if key in self.metadata["EM Dataset"]:
                    del self.metadata["EM Dataset"][key]

            else:
                self.metadata["EM Dataset"][key] = value

        if self.receivers is not None:
            self.receivers.metadata = self.metadata

        if self.transmitters is not None:
            self.transmitters.metadata = self.metadata

    @property
    def inline_offset(self):
        """Inline offset between receiver and transmitter"""
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
    def input_type(self):
        """Data input_type"""
        if "Input type" in self.metadata["EM Dataset"]:
            return self.metadata["EM Dataset"]["Input type"]

        return None

    @input_type.setter
    def input_type(self, value: str):
        input_types = ["Rx", "Tx", "Tx and Rx"]
        assert value in input_types, f"Input 'input_type' must be one of {input_types}"
        self.edit_metadata({"Input type": value})

    @property
    def metadata(self) -> dict:
        """
        Metadata attached to the entity.
        """
        if getattr(self, "_metadata", None) is None:
            metadata = self.workspace.fetch_metadata(self.uid)

            if metadata is None:
                metadata = self.default_metadata
                element = (
                    "Receivers"
                    if "Receivers" in type(self).__name__
                    else "Transmitters"
                )
                metadata["EM Dataset"][element] = self.uid
                self.metadata = metadata
            else:
                self._metadata = metadata

        return self._metadata

    @metadata.setter
    def metadata(self, values: dict):
        if not isinstance(values, dict):
            raise TypeError("'metadata' must be of type 'dict'")

        if "EM Dataset" not in values:
            values = {"EM Dataset": values}

        for key in self.default_metadata["EM Dataset"]:
            if key not in values["EM Dataset"]:
                raise KeyError(f"'{key}' argument missing from the input metadata.")

        for key, value in values["EM Dataset"].items():
            if key in ["Receivers", "Transmitters"] and isinstance(value, str):
                values["EM Dataset"][key] = uuid.UUID(value)

        self._metadata = values
        self.modified_attributes = "metadata"

    @property
    def pitch(self) -> float | uuid.UUID | None:
        """Pitch angle(s) of the transmitter coil"""
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
    def receivers(self):
        """
        The associated TEM receivers
        """
        ...

    @property
    def relative_to_bearing(self):
        """Data relative_to_bearing"""
        if "Angles relative to bearing" in self.metadata["EM Dataset"]:
            return self.metadata["EM Dataset"]["Angles relative to bearing"]

        return None

    @relative_to_bearing.setter
    def relative_to_bearing(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("Input 'relative_to_bearing' must be one of type 'bool'")
        self.edit_metadata({"Angles relative to bearing": value})

    @property
    def roll(self) -> float | uuid.UUID | None:
        """Roll angle(s) of the transmitter coil"""
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
    def timing_mark(self):
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
    def transmitters(self):
        ...

    @property
    def unit(self):
        """Data unit"""
        if "Unit" in self.metadata["EM Dataset"]:
            return self.metadata["EM Dataset"]["Unit"]

        return None

    @unit.setter
    def unit(self, value: str):
        units = [
            "Seconds (s)",
            "Milliseconds (ms)",
            "Microseconds (us)",
            "Nanoseconds (ns)",
        ]
        if value not in units:
            raise ValueError(f"Input 'unit' must be one of {units}")
        self.edit_metadata({"Unit": value})

    @property
    def waveform(self):
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
        """Yaw angle(s) of the transmitter coil"""
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
    _transmitters = None

    def __init__(self, object_type: ObjectType, **kwargs):
        super().__init__(object_type, **kwargs)

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        """
        :return: Default unique identifier
        """
        return cls.__TYPE_UID

    @property
    def receivers(self):
        """
        The associated TEM receivers
        """
        return self

    @property
    def transmitters(self):
        """
        The associated current electrode object (sources).
        """
        if getattr(self, "_transmitters", None) is None:
            if self.metadata is not None:
                tx_uid = self.metadata["EM Dataset"]["Transmitters"]

                try:
                    self.transmitters = self.workspace.get_entity(tx_uid)[0]
                except IndexError:
                    print(
                        "Associated 'AirborneTEMTransmitters' entity not found in Workspace."
                    )
                    return None

        return self._transmitters

    @transmitters.setter
    def transmitters(self, transmitters: AirborneTEMTransmitters):
        if not isinstance(transmitters, AirborneTEMTransmitters):
            raise TypeError(
                f"Provided transmitters must be of type {AirborneTEMTransmitters}. "
                f"{type(transmitters)} provided."
            )
        self._transmitters = transmitters
        self.edit_metadata({"Transmitters": transmitters.uid})


class AirborneTEMTransmitters(BaseAirborneTEM):
    """
    Airborne time-domain electromagnetic transmitters class.
    """

    __TYPE_UID = uuid.UUID("{58c4849f-41e2-4e09-b69b-01cf4286cded}")
    _receivers = None

    def __init__(self, object_type: ObjectType, **kwargs):
        super().__init__(object_type, **kwargs)

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        """
        :return: Default unique identifier
        """
        return cls.__TYPE_UID

    @property
    def transmitters(self):
        """
        The associated current electrode object (sources).
        """
        return self

    @property
    def receivers(self) -> AirborneTEMReceivers | None:
        """
        The associated receivers (receivers)
        """
        if self.metadata is None:
            raise AttributeError("No Current-Receiver metadata set.")

        receivers = self.metadata["EM Dataset"]["Receivers"]

        try:
            return self.workspace.get_entity(receivers)[0]
        except IndexError:
            print("Associated AirborneTEMReceivers entity not found in Workspace.")
            return None

    @receivers.setter
    def receivers(self, receivers: AirborneTEMReceivers):
        if not isinstance(receivers, AirborneTEMReceivers):
            raise TypeError(
                f"Provided receivers must be of type {AirborneTEMReceivers}. "
                f"{type(receivers)} provided."
            )

        self._receivers = receivers
        self.edit_metadata({"Receivers": receivers.uid})
