#  Copyright (c) 2021 Mira Geoscience Ltd.
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

from ...data import Data
from ..curve import Points
from ..object_type import ObjectType


class Magnetotellurics(Points):
    """
    A magnetotellurics survey object.
    """

    __TYPE_UID = uuid.UUID("{b99bd6e5-4fe1-45a5-bd2f-75fc31f91b38}")
    __default_metadata = {
        "EM Dataset": {
            "Channels": [],
            "Input type": "Rx Only",
            "Property groups": [],
            "Receivers": "",
            "Survey type": "Magnetotellurics",
            "Unit": "Hertz (Hz)",
        }
    }
    _input_type = None
    _survey_type = None
    _unit = None

    def __init__(self, object_type: ObjectType, **kwargs):
        super().__init__(object_type, **kwargs)

    @property
    def channels(self):
        """
        List of measured frequencies.
        """
        channels = self.metadata["EM Dataset"]["Channels"]
        return channels

    @channels.setter
    def channels(self, values: list):
        if not isinstance(values, list):
            raise TypeError(
                f"Channel values must be a list of {float}. {type(values)} provided"
            )

        self.metadata["EM Dataset"]["Channels"] = values
        self.modified_attributes = "metadata"

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
    def input_type(self):
        """Type of measurements"""
        if getattr(self, "_input_type", None) is None:
            self._input_type = self.metadata["EM Dataset"]["Input type"]

        return self._input_type

    @property
    def metadata(self) -> dict:
        """
        Metadata attached to the entity.
        """
        if getattr(self, "_metadata", None) is None:
            metadata = self.workspace.fetch_metadata(self.uid)

            if metadata is None:
                self.default_metadata["EM Dataset"]["Receivers"] = str(self.uid)
                metadata = self.default_metadata

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
                raise KeyError(f"{key} argument missing from the input metadata.")

        self._metadata = values
        self.modified_attributes = "metadata"

    @property
    def survey_type(self):
        """Type of EM survey"""
        if getattr(self, "_survey_type", None) is None:
            self._survey_type = self.metadata["EM Dataset"]["Survey type"]

        return self._survey_type

    @property
    def unit(self):
        """Data unit"""
        if getattr(self, "_unit", None) is None:
            self._unit = self.metadata["EM Dataset"]["Unit"]

        return self._unit

    def add_frequency_data(self, data: dict) -> Data | list[Data]:
        """
        Adapted from :func:`~geoh5py.objects.object_base.ObjectBase.add_data` method.

        Add data per component at every frequency defined in
        :attr:`~geoh5py.objects.surveys.magnetotellurics.Magnetotellurics.channels`.
        Data properties such as 'values' and 'entity_type' must be provided as a
        dictionary under each frequency such as:

        .. code-block:: python

            data = {
                "Zxx (real)": {
                    freq_1: {'values': [v_11, v_12, ...]},
                    freq_2: {'values': [v_21, v_22, ...]},
                    },
                },
                "Zxx (imaginary)": {
                    freq_1: {
                        'values': [v_11, v_12, ...],
                        "entity_type": entity_type_A,
                        ...,
                    },
                    freq_2: {...},
                },
            }

        Data values association is always assumed to be 'VERTEX' and name set
        by the component and frequency value.
        A :obj:`geoh5py.groups.property_group.PropertyGroup` for the component gets created
        by default to group all frequencies.

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
                "Input data must be nested dictionaries of component and frequency channels"
            )

        for name, component_block in data.items():
            if not isinstance(component_block, dict):
                raise TypeError(
                    f"Given value to data {name} should of type {dict}. "
                    f"Type {type(component_block)} given instead."
                )

            if len(component_block) != len(self.channels):
                raise ValueError(
                    f"Input component {name} should contain {len(self.channels)} "
                    "frequency values, equal to the number of 'channels'."
                    f"{len(component_block)} values provided."
                )
            for channel in self.channels:
                if channel not in component_block:
                    raise KeyError(
                        f"Channel {channel} Hz is missing from the component {name}."
                    )

                if not isinstance(component_block[channel], dict):
                    raise TypeError(
                        f"Given value to data {channel} should of type {dict}. "
                        f"Type {type(component_block[channel])} given instead."
                    )
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

            prop_group = self.find_or_create_property_group(name=name)
            if prop_group.name not in self.metadata["EM Dataset"]["Property groups"]:
                self.metadata["EM Dataset"]["Property groups"].append(name)
                self.modified_attributes = "metadata"

        self.workspace.finalize()

        return data_objects
