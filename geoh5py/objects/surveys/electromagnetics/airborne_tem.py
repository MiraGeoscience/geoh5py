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

from .base import BaseTEMSurvey


class BaseAirborneTEM(BaseTEMSurvey, Curve):  # pylint: disable=too-many-ancestors
    __INPUT_TYPE = ["Rx", "Tx", "Tx and Rx"]
    _PROPERTY_MAP = {
        "crossline_offset": "Crossline offset",
        "inline_offset": "Inline offset",
        "pitch": "Pitch",
        "roll": "Roll",
        "vertical_offset": "Vertical offset",
        "yaw": "Yaw",
    }

    @property
    def crossline_offset(self) -> float | uuid.UUID | None:
        """
        Numeric value or property UUID for the crossline offset between receiver and transmitter.
        """
        return self.fetch_metadata("crossline_offset")

    @crossline_offset.setter
    def crossline_offset(self, value: float | uuid.UUID | None):
        self.set_metadata("crossline_offset", value)

    def copy(
        self,
        parent=None,
        copy_children: bool = True,
        clear_cache: bool = False,
        mask: np.ndarray | None = None,
        cell_mask: np.ndarray | None = None,
        **kwargs,
    ):
        """
        Sub-class extension of :func:`~geoh5py.objects.cell_object.CellObject.copy`.
        """
        if parent is None:
            parent = self.parent

        omit_list = [
            "_metadata",
            "_receivers",
            "_transmitters",
        ]
        metadata = self.metadata.copy()
        new_entity = super().copy(
            parent=parent,
            clear_cache=clear_cache,
            copy_children=copy_children,
            mask=mask,
            cell_mask=cell_mask,
            omit_list=omit_list,
            **kwargs,
        )

        metadata["EM Dataset"][new_entity.type] = new_entity.uid

        complement: AirborneTEMTransmitters | AirborneTEMReceivers = (
            self.transmitters  # type: ignore
            if isinstance(self, AirborneTEMReceivers)
            else self.receivers
        )
        if complement is not None:
            new_complement = super(Curve, complement).copy(  # type: ignore
                parent=parent,
                omit_list=omit_list,
                copy_children=copy_children,
                clear_cache=clear_cache,
                mask=mask,
            )

            setattr(new_entity, complement.type, new_complement)
            metadata["EM Dataset"][complement.type] = new_complement.uid
            new_complement.metadata = metadata

        new_entity.metadata = metadata

        return new_entity

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
    def default_input_types(self) -> list[str]:
        """Choice of survey creation types."""
        return self.__INPUT_TYPE

    @property
    def default_receiver_type(self):
        """
        :return: Transmitter class
        """
        return AirborneTEMReceivers

    @property
    def default_transmitter_type(self):
        """
        :return: Transmitter class
        """
        return AirborneTEMTransmitters

    def fetch_metadata(self, key: str) -> float | uuid.UUID | None:
        """
        Fetch entry from the metadata.
        """
        field = self._PROPERTY_MAP.get(key, "")
        if field + " value" in self.metadata["EM Dataset"]:
            return self.metadata["EM Dataset"][field + " value"]
        if field + " property" in self.metadata["EM Dataset"]:
            return self.metadata["EM Dataset"][field + " property"]
        return None

    def set_metadata(self, key: str, value: float | uuid.UUID | None):
        if key not in self._PROPERTY_MAP:
            raise ValueError(f"No property map found for key metadata '{key}'.")

        field = self._PROPERTY_MAP[key]
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
    def yaw(self) -> float | uuid.UUID | None:
        """
        Numeric value or property UUID for the yaw angle of the transmitter loop.
        """
        return self.fetch_metadata("yaw")

    @yaw.setter
    def yaw(self, value: float | uuid.UUID):
        self.set_metadata("yaw", value)


class AirborneTEMReceivers(BaseAirborneTEM):  # pylint: disable=too-many-ancestors
    """
    Airborne time-domain electromagnetic receivers class.
    """

    __TYPE_UID = uuid.UUID("{19730589-fd28-4649-9de0-ad47249d9aba}")
    __TYPE = "Receivers"

    def __init__(self, object_type: ObjectType, name="Airborne TEM Rx", **kwargs):
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


class AirborneTEMTransmitters(BaseAirborneTEM):  # pylint: disable=too-many-ancestors
    """
    Airborne time-domain electromagnetic transmitters class.
    """

    __TYPE_UID = uuid.UUID("{58c4849f-41e2-4e09-b69b-01cf4286cded}")
    __TYPE = "Transmitters"

    def __init__(self, object_type: ObjectType, name="Airborne TEM Tx", **kwargs):
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
