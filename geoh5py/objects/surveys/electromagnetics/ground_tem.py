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

from geoh5py.data import ReferencedData
from geoh5py.objects import Curve
from geoh5py.objects.object_base import ObjectType

from .base import BaseTEMSurvey


class BaseGroundTEM(BaseTEMSurvey, Curve):  # pylint: disable=too-many-ancestors
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
                "Tx ID property": None,
                "Unit": "Milliseconds (ms)",
                "Waveform": {"Timing mark": 0.0},
            }
        }

    @property
    def default_receiver_type(self):
        """
        :return: Transmitter class
        """
        return GroundTEMReceiversLargeLoop

    @property
    def default_transmitter_type(self):
        """
        :return: Transmitter class
        """
        return GroundTEMTransmittersLargeLoop


class GroundTEMReceiversLargeLoop(BaseGroundTEM):  # pylint: disable=too-many-ancestors
    """
    Ground time-domain electromagnetic receivers class.
    """

    __TYPE_UID = uuid.UUID("{deebe11a-b57b-4a03-99d6-8f27b25eb2a8}")
    __TYPE = "Receivers"

    _transmitters: GroundTEMTransmittersLargeLoop | None = None

    def __init__(self, object_type: ObjectType, name="Ground TEM Rx", **kwargs):
        self._tx_id_property: ReferencedData | None = None

        super().__init__(object_type, name=name, **kwargs)

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
        return GroundTEMTransmittersLargeLoop

    @property
    def tx_id_property(self) -> ReferencedData | None:
        """
        Default channel units for time or frequency defined on the child class.
        """
        if self._tx_id_property is None:
            if "Tx ID property" in self.metadata["EM Dataset"]:
                data = self.get_data(self.metadata["EM Dataset"]["Tx ID property"])[0]

                if isinstance(data, ReferencedData):
                    self._tx_id_property = data

        return self._tx_id_property

    @tx_id_property.setter
    def tx_id_property(self, value: uuid.UUID | ReferencedData | np.ndarray | None):

        if isinstance(value, uuid.UUID):
            value = self.get_data(value)[0]

        if isinstance(value, np.ndarray):
            if self.transmitters is None or self.transmitters.tx_id_property is None:
                raise AttributeError(
                    "Setting property 'tx_id_property' requires a `transmitters` "
                    "with `tx_id_property` set."
                )

            value = self.add_data(
                {
                    "Transmitter ID": {
                        "values": value,
                        "value_map": self.transmitters.tx_id_property.entity_type.value_map.map,
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
        self.edit_metadata({"Tx ID property": getattr(value, "uid", None)})

    @property
    def type(self):
        """Survey element type"""
        return self.__TYPE


class GroundTEMTransmittersLargeLoop(
    BaseGroundTEM
):  # pylint: disable=too-many-ancestors
    """
    Ground time-domain electromagnetic transmitters class.
    """

    __TYPE_UID = uuid.UUID("{17dbbfbb-3ee4-461c-9f1d-1755144aac90}")
    __TYPE = "Transmitters"

    _receivers: GroundTEMReceiversLargeLoop | None = None

    def __init__(self, object_type: ObjectType, name="Ground TEM Tx", **kwargs):
        self._tx_id_property: ReferencedData | None = None

        super().__init__(object_type, name=name, **kwargs)

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
        return GroundTEMReceiversLargeLoop

    @property
    def tx_id_property(self) -> ReferencedData | None:
        """
        Default channel units for time or frequency defined on the child class.
        """
        if self._tx_id_property is None:
            data = self.get_data("Transmitter ID")[0]

            if isinstance(data, ReferencedData):
                self._tx_id_property = data

        return self._tx_id_property

    @tx_id_property.setter
    def tx_id_property(self, values: np.ndarray):
        if not isinstance(values, np.ndarray):
            raise TypeError("Input value for 'tx_id_property' should be a np.ndarray.)")

        self.add_data(
            {
                "Transmitter ID": {
                    "values": values.astype(np.int32),
                    "value_map": {
                        ind: f"Loop {ind}" for ind in np.unique(values.astype(np.int32))
                    },
                    "type": "referenced",
                }
            }
        )

    @property
    def type(self):
        """Survey element type"""
        return self.__TYPE
