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

# pylint: disable=no-member, too-many-lines
# mypy: disable-error-code="attr-defined"

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING

import numpy as np

from geoh5py.data import ReferencedData

from ... import Curve
from ...object_base import OMIT_LIST, BaseEMSurvey

if TYPE_CHECKING:
    from geoh5py.groups import Group
    from geoh5py.workspace import Workspace


class MovingLoopGroundEMSurvey(BaseEMSurvey, Curve):
    __INPUT_TYPE = ["Rx"]

    @property
    def base_receiver_type(self):
        return Curve

    @property
    def base_transmitter_type(self):
        return Curve

    @property
    def default_input_types(self) -> list[str]:
        """Choice of survey creation types."""
        return self.__INPUT_TYPE

    @property
    def loop_radius(self) -> float | None:
        """Transmitter loop radius"""
        return self.metadata["EM Dataset"].get("Loop radius", None)

    @loop_radius.setter
    def loop_radius(self, value: float | None):
        if not isinstance(value, (float, type(None))):
            raise TypeError("Input 'loop_radius' must be of type 'float'")
        self.edit_metadata({"Loop radius": value})


class LargeLoopGroundEMSurvey(BaseEMSurvey, Curve):
    __INPUT_TYPE = ["Tx and Rx"]
    _tx_id_property: ReferencedData | None = None

    @property
    def base_receiver_type(self):
        return Curve

    @property
    def base_transmitter_type(self):
        return Curve

    def copy_complement(
        self,
        new_entity,
        parent: Group | Workspace | None = None,
        copy_children: bool = True,
        clear_cache: bool = False,
        mask: np.ndarray | None = None,
    ):
        if (
            self.cells is not None
            and new_entity.tx_id_property is None
            and self.tx_id_property is not None
            and self.tx_id_property.values is not None
        ):
            if mask is not None:
                if isinstance(self, self.default_receiver_type):
                    cell_mask = mask
                else:
                    cell_mask = np.all(mask[self.cells], axis=1)
            else:
                cell_mask = np.ones(self.tx_id_property.values.shape[0], dtype=bool)

            new_entity.tx_id_property = self.tx_id_property.values[cell_mask]

        if not (
            new_entity.tx_id_property is not None
            and self.complement is not None
            and self.complement.tx_id_property is not None
            and self.complement.tx_id_property.values is not None
            and self.complement.vertices is not None
            and self.complement.cells is not None
        ):
            return None

        intersect = np.intersect1d(
            new_entity.tx_id_property.values,
            self.complement.tx_id_property.values,
        )

        # Convert cell indices to vertex indices
        if isinstance(
            self.complement,
            self.default_receiver_type,
        ):
            mask = np.r_[
                [(val in intersect) for val in self.complement.tx_id_property.values]
            ]
            tx_ids = self.complement.tx_id_property.values[mask]
        else:
            cell_mask = np.r_[
                [(val in intersect) for val in self.complement.tx_id_property.values]
            ]
            mask = np.zeros(self.complement.vertices.shape[0], dtype=bool)
            mask[self.complement.cells[cell_mask, :]] = True
            tx_ids = self.complement.tx_id_property.values[cell_mask]

        new_complement = (
            self.complement._super_copy(  # pylint: disable=protected-access
                parent=parent,
                omit_list=OMIT_LIST,
                copy_children=copy_children,
                clear_cache=clear_cache,
                mask=mask,
            )
        )

        if isinstance(self, self.default_receiver_type):
            new_entity.transmitters = new_complement
        else:
            new_entity.receivers = new_complement

        if (
            new_complement.tx_id_property is None
            and self.complement.tx_id_property is not None
        ):
            new_complement.tx_id_property = tx_ids

            # Re-number the tx_id_property
            value_map = {
                val: ind
                for ind, val in enumerate(
                    np.r_[0, np.unique(new_entity.transmitters.tx_id_property.values)]
                )
            }
            new_map = {
                val: new_entity.transmitters.tx_id_property.value_map.map[ind]
                for ind, val in value_map.items()
            }
            new_complement.tx_id_property.values = np.asarray(
                [value_map[val] for val in new_complement.tx_id_property.values]
            )
            new_complement.tx_id_property.entity_type.value_map = new_map
            new_entity.tx_id_property.values = np.asarray(
                [value_map[val] for val in new_entity.tx_id_property.values]
            )
            new_entity.tx_id_property.entity_type.value_map = new_map

        return new_complement

    @property
    def default_input_types(self) -> list[str]:
        """Choice of survey creation types."""
        return self.__INPUT_TYPE

    @property
    def tx_id_property(self) -> ReferencedData | None:
        """
        Default channel units for time or frequency defined on the child class.
        """
        if self._tx_id_property is None:
            data = self.get_data("Transmitter ID")
            if any(data) and isinstance(data[0], ReferencedData):
                self._tx_id_property = data[0]

        return self._tx_id_property

    @tx_id_property.setter
    def tx_id_property(self, value: uuid.UUID | ReferencedData | np.ndarray | None):
        if isinstance(value, uuid.UUID):
            value = self.get_data(value)[0]

        if isinstance(value, np.ndarray):
            if (
                self.complement is not None
                and self.complement.tx_id_property is not None
            ):
                entity_type = self.complement.tx_id_property.entity_type
            else:
                value_map = {
                    ind: f"Loop {ind}" for ind in np.unique(value.astype(np.int32))
                }
                value_map[0] = "Unknown"
                entity_type = {  # type: ignore
                    "primitive_type": "REFERENCED",
                    "value_map": value_map,
                }

            value = self.add_data(
                {
                    "Transmitter ID": {
                        "values": value.astype(np.int32),
                        "entity_type": entity_type,
                        "type": "referenced",
                    }
                }
            )

        if not isinstance(value, (ReferencedData, type(None))):
            raise TypeError(
                "Input value for 'tx_id_property' should be of type uuid.UUID, "
                "ReferencedData, np.ndarray or None.)"
            )

        self._tx_id_property = value

        if self.type == "Receivers":
            self.edit_metadata({"Tx ID property": getattr(value, "uid", None)})


class AirborneEMSurvey(BaseEMSurvey, Curve):
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

    @property
    def default_input_types(self) -> list[str]:
        """Choice of survey creation types."""
        return self.__INPUT_TYPE

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


class FEMSurvey(BaseEMSurvey):
    __UNITS = __UNITS = [
        "Hertz (Hz)",
        "KiloHertz (kHz)",
        "MegaHertz (MHz)",
        "Gigahertz (GHz)",
    ]

    @property
    def default_units(self) -> list[str]:
        """
        Accepted frequency units.

        Must be one of "Hertz (Hz)", "KiloHertz (kHz)", "MegaHertz (MHz)", or
        "Gigahertz (GHz)",

        :returns: List of acceptable units for frequency domain channels.
        """
        return self.__UNITS


class TEMSurvey(BaseEMSurvey):
    __UNITS = [
        "Seconds (s)",
        "Milliseconds (ms)",
        "Microseconds (us)",
        "Nanoseconds (ns)",
    ]

    @property
    def default_units(self) -> list[str]:
        """
        Accepted time units.

        Must be one of "Seconds (s)", "Milliseconds (ms)", "Microseconds (us)"
        or "Nanoseconds (ns)"

        :returns: List of acceptable units for time domain channels.
        """
        return self.__UNITS

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

        if self.timing_mark is not None:
            value = self.metadata["EM Dataset"]["Waveform"]
        else:
            value = {"Timing mark": 0.0}

        if isinstance(waveform, np.ndarray):
            if waveform.ndim != 2 or waveform.shape[1] != 2:
                raise ValueError(
                    "Input waveform must be a numpy.ndarray of shape (*, 2)."
                )

            value["Discretization"] = [
                {"current": row[1], "time": row[0]} for row in waveform
            ]

        self.edit_metadata({"Waveform": value})

    @property
    def waveform_parameters(self) -> dict | None:
        """Access the waveform parameters stored as a dictionary."""
        waveform = self.get_data("_waveform_parameters")[0]

        if waveform is not None:
            return json.loads(waveform.values)

        return None
