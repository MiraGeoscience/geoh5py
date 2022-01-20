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

from geoh5py.data import FloatData
from geoh5py.groups import PropertyGroup
from geoh5py.objects import Curve
from geoh5py.objects.object_type import ObjectType


class BaseEMSurvey(Curve):
    """
    A base electromagnetics survey object.
    """

    __METADATA: dict = {"EM Dataset": {}}
    __INPUT_TYPE = None
    __UNITS = None
    _receivers: BaseEMSurvey | None = None
    _transmitters: BaseEMSurvey | None = None

    def __init__(self, object_type: ObjectType, **kwargs):

        super().__init__(object_type, **kwargs)

    def add_component_data(self, data: dict) -> list[PropertyGroup]:
        """
        Add lists of data components to an EM survey. The name of each component is
        appended to the metadata 'Property groups'.

        Data channels must be provided for every frequency or time
        in order specified by
        :attr:`~geoh5py.objects.surveys.electromagnetics.BaseEMSurvey.channels`.
        The data channels can be supplied as either a list of
        :obj:`geoh5py.data.float_data.FloatData` entities or :obj:`uuid.UUID`

        .. code-block:: python

            data = {
                "Component A": [
                    data_entity_1,
                    data_entity_2,
                ],
                "Component B": [...],
            },

        or a nested dictionary of arguments defining new Data entities as defined by the
        :func:`~geoh5py.objects.object_base.ObjectBase.add_data` method.

        .. code-block:: python

            data = {
                "Component A": {
                    time_1: {
                        'values': [v_11, v_12, ...],
                        "entity_type": entity_type_A,
                        ...,
                    },
                    time_2: {...},
                    ...,
                },
                "Component B": {...},
            }

        :param data: Dictionary of data components to be added to the survey.

        :return: List of property groups for all components added.
        """
        prop_groups = []
        if self.channels is None or not self.channels:
            raise AttributeError(
                "The 'channels' attribute of an EMSurvey class must be set before the "
                "'add_component_data' method can be used."
            )

        if not isinstance(data, dict):
            raise TypeError(
                "Input data must be nested dictionaries of components and channels."
            )

        for name, component_block in data.items():
            if name in [pg.name for pg in self.property_groups]:
                raise ValueError(
                    f"PropertyGroup named '{name}' already exists on the survey entity. "
                    f"Consider using the 'edit_metadata' method with "
                    "'Property groups' argument instead."
                )

            if not isinstance(component_block, (dict, list)) or (
                isinstance(component_block, list)
                and not all(isinstance(entry, FloatData) for entry in component_block)
            ):
                raise TypeError(
                    f"List of values provided for component '{name}' must be a list "
                    f"of {FloatData} or {dict} of attributes. "
                    f"Values of type {type(component_block)} provided."
                )

            if len(component_block) != len(self.channels):
                raise ValueError(
                    f"The number of channel values provided must be of len({len(self.channels)}) "
                    "corresponding to the 'channels' attribute. "
                    f"Value of {type(component_block)} and len({len(component_block)}) provided."
                )

            if isinstance(component_block, list):
                assert np.all([entry.parent == self for entry in component_block]), (
                    f"The list of values provided for the component '{name}' "
                    f"must contain {FloatData} belonging to the target survey."
                )

                data_list = component_block

            else:
                data_list = []
                for channel, attr in component_block.items():
                    if not isinstance(attr, dict):
                        raise TypeError(
                            f"Given value to data {channel} should of type {dict} or attributes. "
                            f"Type {type(attr)} given instead."
                        )
                    data_list.append(self.add_data({channel: attr}))

            prop_group = self.add_data_to_group(data_list, name)
            self.edit_metadata({"Property groups": prop_group})
            prop_groups.append(prop_group)
        self.workspace.finalize()

        return prop_groups

    @property
    def channels(self):
        """
        List of measured channels.
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

    @property
    def default_input_types(self) -> list[str] | None:
        """Input types. Must be one of 'Rx', 'Tx', 'Tx and Rx'."""
        return self.__INPUT_TYPE

    @property
    def default_metadata(self):
        """Default metadata structure. Implemented on the child class."""
        return self.__METADATA

    @classmethod
    def default_type_uid(cls):
        """Default unique identifier. Implemented on the child class."""
        ...

    @property
    def default_transmitter_type(self) -> type:
        """
        :return: Transmitters implemented on the child class.
        """
        return type(None)

    @property
    def default_receiver_type(self) -> type:
        """
        :return: Receivers implemented on the child class.
        """
        return type(None)

    @property
    def default_units(self) -> list[str] | None:
        """Accepted time units. Must be one of "Seconds (s)",
        "Milliseconds (ms)", "Microseconds (us)" or "Nanoseconds (ns)"
        """
        return self.__UNITS

    def edit_metadata(self, entries: dict[str, Any]):
        """
        Utility function to edit or add metadata fields and trigger an update
        on the receiver and transmitter entities.

        :param entries: Metadata key value pairs.
        """
        for key, value in entries.items():
            if key in ["Discretization", "Timing mark", "Waveform"]:
                if "Waveform" not in self.metadata["EM Dataset"]:
                    self.metadata["EM Dataset"]["Waveform"] = {}

                wave_key = key.replace("Waveform", "Discretization")
                self.metadata["EM Dataset"]["Waveform"][wave_key] = value

            elif key == "Property groups":
                self._edit_validate_property_groups(value)

            elif value is None:
                if key in self.metadata["EM Dataset"]:
                    del self.metadata["EM Dataset"][key]

            else:
                self.metadata["EM Dataset"][key] = value

        if getattr(self, "receivers", None) is not None:
            getattr(self, "receivers").metadata = self.metadata

        if getattr(self, "transmitters", None) is not None:
            getattr(self, "transmitters").metadata = self.metadata

    def _edit_validate_property_groups(self, value: str | PropertyGroup | list):
        """
        Add or append property groups to the metadata.

        :param value:
        """
        if value is None:
            self.metadata["EM Dataset"]["Property groups"] = []

        if not isinstance(value, list):
            value = [value]

        for val in value:
            if isinstance(val, PropertyGroup):
                prop_group = val
            elif isinstance(val, str) and val in [
                pg.name for pg in self.property_groups
            ]:
                prop_group = self.find_or_create_property_group(name=val)
            else:
                raise TypeError(
                    "Property group must be a list of existing PropertyGroup "
                    + "or PropertyGroup names."
                )

            if len(prop_group.properties) != len(self.channels):
                raise ValueError(
                    f"Number of properties in group '{prop_group.name}' "
                    + "differ from the number of 'channels'."
                )

            if prop_group.name not in self.metadata["EM Dataset"]["Property groups"]:
                self.metadata["EM Dataset"]["Property groups"].append(prop_group.name)

    @property
    def input_type(self):
        """Data input type. Must be one of 'Rx', 'Tx' or 'Tx and Rx'"""
        if "Input type" in self.metadata["EM Dataset"]:
            return self.metadata["EM Dataset"]["Input type"]

        return None

    @input_type.setter
    def input_type(self, value: str):
        if self.default_input_types is not None:
            assert (
                value in self.default_input_types
            ), f"Input 'input_type' must be one of {self.default_input_types}"
            self.edit_metadata({"Input type": value})

    @property
    def metadata(self) -> dict:
        """Metadata attached to the entity."""
        if getattr(self, "_metadata", None) is None:
            metadata = self.workspace.fetch_metadata(self.uid)

            if metadata is None:
                metadata = self.default_metadata
                for name in ["Receivers", "Transmitters"]:
                    if name in type(self).__name__:
                        metadata["EM Dataset"][name] = self.uid
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
    def receivers(self) -> BaseEMSurvey | None:
        """
        The associated TEM receivers.
        """
        if getattr(self, "_receivers", None) is None:
            if self.metadata is not None and "Receivers" in self.metadata["EM Dataset"]:
                rx_uid = self.metadata["EM Dataset"]["Receivers"]

                try:
                    self._receivers = self.workspace.get_entity(rx_uid)[0]
                except IndexError:
                    print("Associated Receivers entity not found in Workspace.")
                    return None

        return self._receivers

    @receivers.setter
    def receivers(self, receivers: BaseEMSurvey):
        if isinstance(self.default_receiver_type, type(None)):
            raise NotImplementedError(f"EM class {self} does not contain receivers.")

        if not isinstance(receivers, self.default_receiver_type):
            raise TypeError(
                f"Provided receivers must be of type {self.default_receiver_type}. "
                f"{type(receivers)} provided."
            )
        self._receivers = receivers
        self.edit_metadata({"Receivers": receivers.uid})

    @property
    def survey_type(self):
        """Data input type. Must be one of 'Rx', 'Tx' or 'Tx and Rx'"""
        if "Survey type" in self.metadata["EM Dataset"]:
            return self.metadata["EM Dataset"]["Survey type"]

        return None

    @property
    def transmitters(self) -> BaseEMSurvey | None:
        """
        The associated TEM transmitters (sources).
        """
        if getattr(self, "_transmitters", None) is None:
            if (
                self.metadata is not None
                and "Transmitters" in self.metadata["EM Dataset"]
            ):
                tx_uid = self.metadata["EM Dataset"]["Transmitters"]

                try:
                    self._transmitters = self.workspace.get_entity(tx_uid)[0]
                except IndexError:
                    print("Associated transmitters entity not found in Workspace.")
                    return None

        return self._transmitters

    @transmitters.setter
    def transmitters(self, transmitters: BaseEMSurvey):
        if isinstance(self.default_transmitter_type, type(None)):
            raise NotImplementedError(f"EM class {self} does not contain transmitters.")

        if not isinstance(transmitters, self.default_transmitter_type):
            raise TypeError(
                f"Provided transmitters must be of type {self.default_transmitter_type}. "
                f"{type(transmitters)} provided."
            )
        self._transmitters = transmitters
        self.edit_metadata({"Transmitters": transmitters.uid})

    @property
    def unit(self) -> float | None:
        """
        Default channel units for time or frequency defined on the child class.
        """
        if "Unit" in self.metadata["EM Dataset"]:
            return self.metadata["EM Dataset"]["Unit"]

        return None

    @unit.setter
    def unit(self, value: str):
        if self.default_units is not None:
            if value not in self.default_units:
                raise ValueError(f"Input 'unit' must be one of {self.default_units}")
            self.edit_metadata({"Unit": value})
