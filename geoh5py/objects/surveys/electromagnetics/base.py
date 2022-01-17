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

from typing import Any

import numpy as np

from geoh5py.data import Data
from geoh5py.groups import PropertyGroup
from geoh5py.objects import Curve
from geoh5py.objects.object_type import ObjectType


class BaseEMSurvey(Curve):
    """
    A base electromagnetics survey object.
    """

    def __init__(self, object_type: ObjectType, **kwargs):
        super().__init__(object_type, **kwargs)

    def add_component_data(self, data: dict) -> Data | list[Data]:
        """
        Add data per component at every channel as defined in
        :attr:`~geoh5py.objects.surveys.electromagnetics.BaseEMSurvey.channels`.

        Data channels must be provided for every frequencies | times
        as data entities or as dictionaries of arguments following the same
        convention as the :func:`~geoh5py.objects.object_base.ObjectBase.add_data`
        method.

        .. code-block:: python

            data = {
                "Zxx (real)": {
                    freq_1: data_entity_1,
                    freq_2: data_entity_2,
                    },
                ...,
            },

        or similarly

        .. code-block:: python

            data = {
                "dBzdt": {
                    time_1: {
                        'values': [v_11, v_12, ...],
                        "entity_type": entity_type_A,
                        ...,
                    },
                    time_2: {...},
                    ...,
                },
            }

        Data values association is always assumed to be 'VERTEX' and name set
        by the component and channel value.

        A :obj:`geoh5py.groups.property_group.PropertyGroup` for the component gets created
        by default to group all channels.

        :param data: Dictionary of data to be added to the object

        :return: List of new Data objects.
        """
        data_objects = []
        if self.channels is None or not self.channels:
            raise AttributeError(
                "The 'channels' property defining frequencies must be set before adding data."
            )

        if not isinstance(data, dict):
            raise TypeError(
                "Input data must be nested dictionaries of component and channels "
                + "(frequencies or times)"
            )

        for name, component_block in data.items():
            if isinstance(component_block, PropertyGroup):
                self.edit_metadata({"Property groups": component_block})
                continue

            if not isinstance(component_block, dict):
                raise TypeError(
                    f"Given value to data {name} should of type {dict} or {PropertyGroup}. "
                    f"Type {type(component_block)} given instead."
                )

            if len(component_block) != len(self.channels):
                raise ValueError(
                    f"Input component {name} should contain {len(self.channels)} "
                    "channel values, equal to the number of 'channels' (frequencies)."
                    f"{len(component_block)} values provided."
                )
            for channel in self.channels:
                if channel not in component_block:
                    raise KeyError(
                        f"Channel {channel} is missing from the component {name}."
                    )

                if isinstance(component_block[channel], dict):
                    component_block[channel]["name"] = name + f" {channel: .3e}"
                    entity_type = self.validate_data_type(component_block[channel])
                    kwargs = {"parent": self, "association": "VERTEX"}
                    for key, val in component_block[channel].items():
                        if key in ["parent", "association", "entity_type", "type"]:
                            continue
                        kwargs[key] = val

                    data_object = self.workspace.create_entity(
                        Data, entity=kwargs, entity_type=entity_type
                    )
                    self.add_data_to_group(data_object, name)

                    data_objects.append(data_object)
                elif isinstance(component_block[channel], Data):
                    self.add_data_to_group(component_block[channel], name)
                    data_objects.append(component_block[channel])
                else:
                    raise TypeError(
                        f"Given value to data {channel} should of type {dict} or {Data}. "
                        f"Type {type(component_block[channel])} given instead."
                    )

            prop_group = self.find_or_create_property_group(name=name)
            self.edit_metadata({"Property groups": prop_group})

        self.workspace.finalize()

        return data_objects

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
    def default_input_types(self) -> list[str]:
        """Accepted input types. Implemented on the child class."""
        ...

    @property
    def default_metadata(self):
        """Default metadata structure. Implemented on the child class."""
        ...

    @classmethod
    def default_type_uid(cls):
        """Default unique identifier. Implemented on the child class."""
        ...

    @property
    def default_units(self) -> list[str]:
        """Accepted channel units. Implemented on the child class."""
        ...

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
                self.edit_property_groups(value)

            elif value is None:
                if key in self.metadata["EM Dataset"]:
                    del self.metadata["EM Dataset"][key]

            else:
                self.metadata["EM Dataset"][key] = value

        if getattr(self, "receivers", None) is not None:
            self.receivers.metadata = self.metadata

        if getattr(self, "transmitters", None) is not None:
            self.transmitters.metadata = self.metadata

    def edit_property_groups(self, value: str | PropertyGroup | list):
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
                    "Property group must be a list of existing PropertyGroups "
                    + "or known property_group names."
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
        assert (
            value in self.default_input_types
        ), f"Input 'input_type' must be one of {self.default_input_types}"
        self.edit_metadata({"Input type": value})

    @property
    def metadata(self) -> dict:
        """Metadata attached to the entity. Must be implemented by the child class."""
        ...

    @metadata.setter
    def metadata(self, values: dict):
        raise NotImplementedError(
            f"Setter of 'metadata' for {values} must be "
            "implemented on the child class."
        )

    @property
    def receivers(self):
        """
        The associated EM receivers. Implemented on the child class.
        """
        ...

    @property
    def survey_type(self):
        """Data input type. Must be one of 'Rx', 'Tx' or 'Tx and Rx'"""
        if "Survey type" in self.metadata["EM Dataset"]:
            return self.metadata["EM Dataset"]["Survey type"]

        return None

    @property
    def transmitters(self):
        """
        The associated EM transmitters. Implemented on the child class.
        """
        ...

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
        if value not in self.default_units:
            raise ValueError(f"Input 'unit' must be one of {self.default_units}")
        self.edit_metadata({"Unit": value})
