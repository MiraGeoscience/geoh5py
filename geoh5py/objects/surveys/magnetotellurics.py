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
from ...groups import PropertyGroup
from ..curve import Points
from ..object_type import ObjectType


class Magnetotellurics(Points):
    """
    A magnetotelluric survey object.
    """

    __TYPE_UID = uuid.UUID("{b99bd6e5-4fe1-45a5-bd2f-75fc31f91b38}")
    _input_type = "Rx Only"
    _survey_type = "Magnetotellurics"
    _unit = "Hertz (Hz)"
    _default_metadata = {
        "EM Dataset": {
            "Channels": [],
            "Input type": _input_type,
            "Property groups": [],
            "Receivers": "",
            "Survey type": _survey_type,
            "Unit": _unit,
        }
    }

    def __init__(self, object_type: ObjectType, **kwargs):
        super().__init__(object_type, **kwargs)

    @property
    def channels(self):
        """
        List of measured frequencies.
        """
        if self.metadata is None:
            raise AttributeError("No metadata set.")

        channels = self.metadata["EM Dataset"]["Channels"]
        return channels

    @channels.setter
    def channels(self, values: list):
        if not isinstance(values, list):
            raise ValueError(
                f"Channel values must be a list of {float}. {type(values)} provided"
            )

        if self.metadata is None:
            raise AttributeError("No metadata set.")

        self.metadata["EM Dataset"]["Channels"] = values
        self.modified_attributes = "metadata"

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        """
        :return: Default unique identifier
        """
        return cls.__TYPE_UID

    @property
    def input_type(self):
        """Type of measurements"""
        return self._input_type

    @property
    def metadata(self) -> dict:
        """
        Metadata attached to the entity.
        """
        if getattr(self, "_metadata", None) is None:
            metadata = self.workspace.fetch_metadata(self.uid)

            if metadata is None:
                self._default_metadata["EM Dataset"]["Receivers"] = str(self.uid)
                metadata = self._default_metadata

            self._metadata = metadata
        return self._metadata

    @metadata.setter
    def metadata(self, values: dict):

        if "EM Dataset" not in values.keys():
            raise KeyError("'EM Dataset' must be a 'metadata' key")

        for key in self._default_metadata:
            if key not in values:
                raise KeyError(f"{key} argument missing from the input metadata.")

        self._metadata = values
        self.modified_attributes = "metadata"

    @property
    def property_groups(self) -> list[PropertyGroup]:
        """
        :obj:`list` of :obj:`~geoh5py.groups.property_group.PropertyGroup`.
        """
        return self._property_groups

    @property_groups.setter
    def property_groups(self, prop_groups: list[PropertyGroup]):
        # Check for existing property_group
        for prop_group in prop_groups:
            if not any(
                pg.uid == prop_group.uid for pg in self.property_groups
            ) and not any(pg.name == prop_group.name for pg in self.property_groups):
                prop_group.parent = self

                self.modified_attributes = "property_groups"
                self._property_groups = self.property_groups + [prop_group]

        if self.property_groups is not None:
            self.metadata["EM dataset"]["Property groups"] = [
                pg.name for pg in self.property_groups
            ]
        self.modified_attributes = "metadata"

    @property
    def survey_type(self):
        """Type of EM survey"""
        return self._survey_type

    @property
    def unit(self):
        """Data unit"""
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
        if self.channels is None:
            raise AttributeError(
                "The 'channels' property defining frequencies must be set before adding data."
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

        self.workspace.finalize()

        if len(data_objects) == 1:
            return data_object

        return data_objects
