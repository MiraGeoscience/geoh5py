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

from ..curve import Curve
from ..object_type import ObjectType


class Receivers(Curve):
    """
    Airborne time-domain electromagnetic receivers class.
    """

    __TYPE_UID = uuid.UUID("{19730589-fd28-4649-9de0-ad47249d9aba}")
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
    def channels(self, values: list):
        if not isinstance(values, list) and np.all(
            [isinstance(x, float) for x in values]
        ):
            raise TypeError(
                f"Channel values must be a list of {float}. {type(values)} provided"
            )

        self.metadata["EM Dataset"]["Channels"] = values
        self.modified_attributes = "metadata"

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
    def default_metadata(self) -> dict:
        """
        :return: Default unique identifier
        """
        return self.__default_metadata.copy()

    @property
    def inline_offset(self):
        """Inline offset between receiver and transmitter"""
        if "Inline offset value" in self.metadata["EM Dataset"]:
            return self.metadata["EM Dataset"]["Inline offset value"]
        if "Inline offset property" in self.metadata["EM Dataset"]:
            return self.metadata["EM Dataset"]["Inline offset property"]
        return None

    @inline_offset.setter
    def inline_offset(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("Input 'inline_offset' must be one of type 'bool'")
        self.metadata["EM Dataset"]["Angles relative to bearing"] = value
        self.modified_attributes = "metadata"

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
        self.metadata["EM Dataset"]["Input type"] = value
        self.modified_attributes = "metadata"

    @property
    def metadata(self) -> dict:
        """
        Metadata attached to the entity.
        """
        if getattr(self, "_metadata", None) is None:
            metadata = self.workspace.fetch_metadata(self.uid)

            if metadata is None:
                metadata = self.default_metadata
                metadata["EM Dataset"]["Receivers"] = str(self.uid)

            self._metadata = metadata
        return self._metadata

    @metadata.setter
    def metadata(self, values: dict):

        if not isinstance(values, dict):
            raise TypeError("'metadata' must be of type 'dict'")

        if "EM Dataset" not in values:
            raise KeyError("'EM Dataset' must be a 'metadata' key")

        for key in self.default_metadata["EM Dataset"]:
            if key not in values["EM Dataset"]:
                raise KeyError(f"'{key}' argument missing from the input metadata.")

        known_keys = [
            "Angles relative to bearing",
            "Inline offset value",
            "Inline offset property",
            "Input type",
            "Loop radius",
            "Pitch value",
            "Pitch property",
            "Property groups",
            "Receivers",
            "Roll value",
            "Roll property",
            "Survey type",
            "Table values",
            "Transmitters",
            "Unit",
            "Vertical offset value",
            "Vertical offset property",
            "Yaw value",
            "Yaw property",
        ]

        for key in values["EM Dataset"]:
            if key not in known_keys:
                raise ValueError(f"Input metadata {key} is not a known key.")

        self._metadata = values
        self.modified_attributes = "metadata"

    @property
    def receivers(self):
        """
        The associated tem receivers
        """
        return self

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
        self.metadata["EM Dataset"]["Angles relative to bearing"] = value
        self.modified_attributes = "metadata"

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

        self.metadata["EM Dataset"]["Waveform"]["Timing mark"] = timing_mark
        self.modified_attributes = "metadata"

    @property
    def transmitters(self):
        """
        The associated current electrode object (sources).
        """
        if self.metadata is None:
            raise AttributeError("No Current-Receiver metadata set.")
        currents = self.metadata["EM Dataset"]["Transmitters"]

        try:
            return self.workspace.get_entity(currents)[0]
        except IndexError:
            print("Associated Transmitters entity not found in Workspace.")
            return None

    @transmitters.setter
    def transmitters(self, transmitters: Transmitters):
        if not isinstance(transmitters, Transmitters):
            raise TypeError(
                f"Provided transmitters must be of type {Transmitters}. "
                f"{type(transmitters)} provided."
            )
        self.metadata["EM Dataset"]["Transmitters"] = transmitters.uid
        self.modified_attributes = "metadata"

    @property
    def unit(self):
        """Data unit"""
        if "Unit" in self.metadata["EM Dataset"]:
            return self.metadata["EM Dataset"]["Unit"]

        return None

    @unit.setter
    def unit(self, value: str):
        units = [
            "s",
            "ms",
            "us",
            "ns",
        ]
        assert value in units, f"Input 'unit' must be one of {units}"
        self.metadata["EM Dataset"]["Unit"] = value
        self.modified_attributes = "metadata"

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

        if waveform is None:
            if "Waveform" in self.metadata["EM Dataset"]:
                del self.metadata["EM Dataset"]["Waveform"]

        elif isinstance(waveform, np.ndarray):
            if waveform.ndim != 2 or waveform.shape[1] != 2:
                raise ValueError(
                    "Input waveform must be a numpy.ndarray of shape (*, 2)."
                )

            self.metadata["EM Dataset"]["Waveform"]["Discretization"] = {
                {"time": row[0], "current": row[1]} for row in waveform
            }
        else:
            raise ValueError("Input waveform must be a numpy.ndarray or None.")
        self.modified_attributes = "metadata"


class Transmitters(Receivers):
    """
    Airborne time-domain electromagnetic transmitters class.
    """

    __TYPE_UID = uuid.UUID("{58c4849f-41e2-4e09-b69b-01cf4286cded}")

    def __init__(self, object_type: ObjectType, **kwargs):
        self._current_line_id: uuid.UUID | None
        self._metadata: dict[uuid.UUID, uuid.UUID] | None

        super().__init__(object_type, **kwargs)

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        """
        :return: Default unique identifier
        """
        return cls.__TYPE_UID

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

        new_receivers = parent.workspace.copy_to_parent(
            self.receivers,
            parent,
            copy_children=copy_children,
            omit_list=omit_list,
        )
        new_entity.receivers = new_receivers
        parent.workspace.finalize()

        return new_entity

    @property
    def transmitters(self):
        """
        The associated current electrode object (sources).
        """
        return self

    @transmitters.setter
    def transmitters(self, _):
        ...

    @property
    def receivers(self) -> Receivers | None:
        """
        The associated receivers (receivers)
        """
        if self.metadata is None:
            raise AttributeError("No Current-Receiver metadata set.")

        receivers = self.metadata["EM Dataset"]["Receivers"]

        try:
            return self.workspace.get_entity(receivers)[0]
        except IndexError:
            print("Associated Receivers entity not found in Workspace.")
            return None

    @receivers.setter
    def receivers(self, receivers: Receivers):
        if not isinstance(receivers, Receivers):
            raise TypeError(
                f"Provided receivers must be of type {Receivers}. "
                f"{type(receivers)} provided."
            )

        self.metadata["EM Dataset"]["Receivers"] = receivers.uid
        self.modified_attributes = "metadata"
