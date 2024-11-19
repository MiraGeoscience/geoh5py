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

# pylint: disable=no-member, too-many-lines, too-many-ancestors
# mypy: disable-error-code="attr-defined"

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any
from warnings import warn

import numpy as np

from geoh5py.data import FloatData, IntegerData, ReferencedData
from geoh5py.groups.property_group import PropertyGroup
from geoh5py.objects import Curve
from geoh5py.objects.object_base import ObjectBase
from geoh5py.shared.utils import str2uuid, str_json_to_dict


if TYPE_CHECKING:
    from geoh5py.groups import Group
    from geoh5py.workspace import Workspace

TYPE_MAP = {
    "Transmitters": "transmitters",
    "Receivers": "receivers",
    "Base stations": "base_stations",
}
OMIT_LIST = [
    "_receivers",
    "_transmitters",
    "_base_stations",
    "_tx_id_property",
    "_metadata",
]


class BaseEMSurvey(ObjectBase, ABC):  # pylint: disable=too-many-public-methods
    """
    A base electromagnetics survey object.
    """

    __INPUT_TYPE = None
    __TYPE = None
    __UNITS = None

    def __init__(self, **kwargs):
        self._receivers: BaseEMSurvey | None = None
        self._transmitters: BaseEMSurvey | None = None
        self._tx_id_property: ReferencedData | IntegerData | None = None

        super().__init__(**kwargs)

    def add_components_data(self, data: dict) -> list[PropertyGroup]:
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
                "'add_components_data' method can be used."
            )

        if not isinstance(data, dict):
            raise TypeError(
                "Input data must be nested dictionaries of components and channels."
            )

        for name, data_block in data.items():
            prop_group = self.add_validate_component_data(name, data_block)
            prop_groups.append(prop_group)

        return prop_groups

    def add_validate_component_data(self, name: str, data_block: list | dict):
        """
        Append a property group to the entity and its metadata after validations.
        """
        if self.property_groups is not None and name in [
            pg.name for pg in self.property_groups
        ]:
            raise ValueError(
                f"PropertyGroup named '{name}' already exists on the survey entity. "
                f"Consider using the 'edit_em_metadata' method with "
                "'Property groups' argument instead."
            )

        if not isinstance(data_block, (dict, list)) or (
            isinstance(data_block, list)
            and not all(isinstance(entry, FloatData) for entry in data_block)
        ):
            raise TypeError(
                f"List of values provided for component '{name}' must be a list "
                f"of {FloatData} or {dict} of attributes. "
                f"Values of type {type(data_block)} provided."
            )

        if len(data_block) != len(self.channels):
            raise ValueError(
                f"The number of channel values provided must be of len({len(self.channels)}) "
                "corresponding to the 'channels' attribute. "
                f"Value of {type(data_block)} and len({len(data_block)}) provided."
            )

        if isinstance(data_block, list):
            if not np.all([entry.parent == self for entry in data_block]):
                raise ValueError(
                    f"The list of values provided for the component '{name}' "
                    f"must contain {FloatData} belonging to the target survey."
                )

            data_list = data_block

        else:
            data_list = []
            for channel, attr in data_block.items():
                if not isinstance(attr, dict):
                    raise TypeError(
                        f"Given value to data {channel} should of type {dict} or attributes. "
                        f"Type {type(attr)} given instead."
                    )
                data_list.append(self.add_data({channel: attr}))

        prop_group = self.add_data_to_group(data_list, name)
        self.edit_em_metadata({"Property groups": prop_group})

        return prop_group

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

        self.edit_em_metadata({"Channels": values})

    @property
    def complement(self) -> BaseEMSurvey | None:
        """Returns the complement object for self."""
        return None

    @property
    def components(self) -> dict | None:
        """
        Rapid access to the list of data entities for all components.
        """
        if "Property groups" in self.metadata["EM Dataset"]:
            components = {}
            for name in self.metadata["EM Dataset"]["Property groups"]:
                prop_group = self.fetch_property_group(name=name)

                if prop_group.properties is None:
                    continue

                components[name] = [
                    self.workspace.get_entity(uid)[0] for uid in prop_group.properties
                ]
            return components

        return None

    def copy(  # pylint: disable=too-many-arguments
        self,
        parent: Group | Workspace | None = None,
        *,
        copy_children: bool = True,
        clear_cache: bool = False,
        mask: np.ndarray | None = None,
        cell_mask: np.ndarray | None = None,
        copy_complement: bool = True,
        **kwargs,
    ):
        """
        Sub-class extension of :func:`~geoh5py.objects.cell_object.CellObject.copy`.
        """
        if parent is None:
            parent = self.parent

        new_entity = super().copy(
            parent=parent,
            clear_cache=clear_cache,
            copy_children=copy_children,
            mask=mask,
            cell_mask=cell_mask,
            omit_list=OMIT_LIST,
            **kwargs,
        )

        # Copy metadata except reference to entities UUID
        for key, value in self.metadata["EM Dataset"].items():
            if not isinstance(value, (uuid.UUID, type(None))):
                new_entity.edit_em_metadata({key: value})

        if copy_complement:
            self.copy_complement(
                new_entity,
                parent=parent,
                copy_children=copy_children,
                clear_cache=clear_cache,
                mask=mask,
            )

        return new_entity

    def copy_complement(
        self,
        new_entity,
        *,
        parent: Group | Workspace | None = None,
        copy_children: bool = True,
        clear_cache: bool = False,
        mask: np.ndarray | None = None,
    ) -> BaseEMSurvey | None:
        """
        Copy the complement entity to the new entity.

        :param new_entity: New entity to copy the complement to.
        :param parent: Parent group or workspace.
        :param copy_children: Copy children entities.
        :param clear_cache: Clear the cache.
        :param mask: Mask on vertices to apply to the data.
        """
        if self.complement is None:
            return None

        # Reset the mask based on Tx ID if it exists
        if (
            new_entity.tx_id_property is not None
            and self.complement.tx_id_property is not None
            and self.complement.tx_id_property.values is not None
            and mask is not None
        ):
            max_id = np.max(
                [
                    self.complement.tx_id_property.values.max(),
                    new_entity.tx_id_property.values.max(),
                ]
            )
            ids_mask = np.zeros(max_id + 1, dtype=bool)
            ids_mask[new_entity.tx_id_property.values] = True
            mask = ids_mask[self.complement.tx_id_property.values]

        new_complement = self.complement._super_copy(  # pylint: disable=protected-access
            parent=parent,
            omit_list=OMIT_LIST,
            copy_children=copy_children,
            clear_cache=clear_cache,
            mask=mask,
        )

        setattr(
            new_entity,
            TYPE_MAP[self.complement.type],  # pylint: disable=no-member
            new_complement,
        )

        return new_complement

    @property
    @abstractmethod
    def default_input_types(self) -> list[str] | None:
        """
        Input types.

        Must be one of 'Rx', 'Tx', 'Tx and Rx', 'Rx only', 'Rx and base stations'."""

    @property
    def default_metadata(self):
        """Default metadata structure. Implemented on the child class."""
        return {"EM Dataset": {}}

    @property
    @abstractmethod
    def default_transmitter_type(self) -> type:
        """
        :return: Transmitters implemented on the child class.
        """

    @property
    @abstractmethod
    def default_receiver_type(self) -> type:
        """
        :return: Receivers implemented on the child class.
        """

    @property
    @abstractmethod
    def default_units(self) -> list[str]:
        """
        List of accepted units.
        """

    def edit_em_metadata(self, entries: dict[str, Any]):
        """
        Utility function to edit or add metadata fields and trigger an update
        on the receiver and transmitter entities.

        :param entries: Metadata key value pairs.
        """
        em_metadata = self.metadata.get("EM Dataset", {})

        for key, value in entries.items():
            if key == "Property groups":
                self._edit_validate_property_groups(value)
            elif value is None:
                if key in em_metadata:
                    del em_metadata[key]
            else:
                em_metadata[key] = value

        self.metadata = {"EM Dataset": em_metadata}

    def edit_metadata(self, entries: dict[str, Any]):
        """
        WILL BE DEPRECATED IN 0.10 version.
        The name will change to edit_em_metadata.

        :param entries:
        :return:
        """
        warn(
            "DEPRECATION WARNING"
            "The method 'edit_metadata' will be deprecated in 0.10 version. "
            "It will be replaced by 'edit_em_metadata'"
        )
        self.edit_em_metadata(entries)

    @property
    def input_type(self) -> str | None:
        """Data input type. Must be one of 'Rx', 'Tx' or 'Tx and Rx'"""
        if "Input type" in self.metadata["EM Dataset"]:
            return self.metadata["EM Dataset"]["Input type"]

        return None

    @input_type.setter
    def input_type(self, value: str):
        if self.default_input_types is None:
            return

        if value not in self.default_input_types:
            raise ValueError(
                "Input 'input_type' must be one of "
                f"{self.default_input_types}. {value} provided."
            )
        self.edit_em_metadata({"Input type": value})

    @property
    def metadata(self):
        """Metadata attached to the entity."""
        if getattr(self, "_metadata", None) is None:
            metadata = self.workspace.fetch_metadata(self)
            self._metadata = self.validate_em_metadata(metadata)

        return self._metadata

    @metadata.setter
    def metadata(self, values: dict | np.ndarray | bytes | None):
        self._metadata = self.validate_em_metadata(values)

        if self.on_file:
            self.workspace.update_attribute(self, "metadata")

            if self.complement is not None:
                self.complement._metadata = self._metadata  # pylint: disable=protected-access

                if self.complement.on_file:
                    self.workspace.update_attribute(self.complement, "metadata")

    @property
    def receivers(self) -> BaseEMSurvey | None:
        """
        The associated TEM receivers.
        """
        if getattr(self, "_receivers", None) is None:
            if self.metadata is not None and "Receivers" in self.metadata["EM Dataset"]:
                receiver = self.metadata["EM Dataset"]["Receivers"]
                receiver_entity = self.workspace.get_entity(receiver)[0]

                if isinstance(receiver_entity, BaseEMSurvey):
                    self._receivers = receiver_entity

        return self._receivers

    @receivers.setter
    def receivers(self, receivers: BaseEMSurvey):
        if not isinstance(receivers, self.default_receiver_type):
            raise TypeError(
                f"Provided receivers must be of type {self.default_receiver_type}. "
                f"{type(receivers)} provided."
            )

        if receivers.tx_id_property is not None:
            self.edit_em_metadata({"Tx ID property": receivers.tx_id_property.uid})

            if isinstance(
                self.tx_id_property, ReferencedData | IntegerData
            ) and isinstance(receivers.tx_id_property, ReferencedData | IntegerData):
                self.tx_id_property.entity_type = receivers.tx_id_property.entity_type

        self._receivers = receivers
        self.edit_em_metadata({"Receivers": receivers.uid})

    @property
    def survey_type(self) -> str | None:
        """Data input type. Must be one of 'Rx', 'Tx' or 'Tx and Rx'"""
        if "Survey type" in self.metadata["EM Dataset"]:
            return self.metadata["EM Dataset"]["Survey type"]

        return None

    @property
    def transmitters(self):
        """
        The associated TEM transmitters (sources).
        """
        if getattr(self, "_transmitters", None) is None:
            if (
                self.metadata is not None
                and "Transmitters" in self.metadata["EM Dataset"]
            ):
                transmitter = self.metadata["EM Dataset"]["Transmitters"]
                transmitter_entity = self.workspace.get_entity(transmitter)[0]

                if isinstance(transmitter_entity, BaseEMSurvey):
                    self._transmitters = transmitter_entity

        return self._transmitters

    @transmitters.setter
    def transmitters(self, transmitters: BaseEMSurvey):
        if isinstance(None, self.default_transmitter_type):
            raise AttributeError(
                f"The 'transmitters' attribute cannot be set on class {type(self)}."
            )

        if not isinstance(transmitters, self.default_transmitter_type):
            raise TypeError(
                f"Provided transmitters must be of type {self.default_transmitter_type}. "
                f"{type(transmitters)} provided."
            )

        if transmitters.tx_id_property is not None:
            self.edit_em_metadata(
                {"Tx ID tx property": transmitters.tx_id_property.uid}
            )
            if isinstance(
                self.tx_id_property, ReferencedData | IntegerData
            ) and isinstance(transmitters.tx_id_property, ReferencedData | IntegerData):
                self.tx_id_property.entity_type = (
                    transmitters.tx_id_property.entity_type
                )

        self._transmitters = transmitters
        self.edit_em_metadata({"Transmitters": transmitters.uid})

    @property
    @abstractmethod
    def type(self):
        """Survey element type"""

    @property
    def unit(self) -> float | None:
        """
        Default channel units for time or frequency defined on the child class.
        """
        return self.metadata["EM Dataset"].get("Unit")

    @unit.setter
    def unit(self, value: str):
        if self.default_units is not None:
            if value not in self.default_units:
                raise ValueError(f"Input 'unit' must be one of {self.default_units}")
            self.edit_em_metadata({"Unit": value})

    def _edit_validate_property_groups(
        self, values: PropertyGroup | list[PropertyGroup] | list[str] | None
    ):
        """
        Add or append property groups to the metadata.

        :param value:
        """
        if not values:
            self.metadata["EM Dataset"]["Property groups"] = []
            return

        if not isinstance(values, list):
            values = [values]

        groups = (
            {group.name: group for group in self.property_groups}
            if self.property_groups
            else {}
        )

        for value in values:
            if self.property_groups is None:
                continue

            if not isinstance(value, (PropertyGroup, str)):
                raise TypeError(
                    "Input value for 'Property groups' must be a PropertyGroup or "
                    "name of an existing PropertyGroup."
                )

            if not (value in groups or value in groups.values()):
                raise ValueError("Property group must be an existing PropertyGroup.")

            if isinstance(value, str):
                value = groups[value]

            if value.properties is not None and len(value.properties) != len(
                self.channels
            ):
                raise ValueError(
                    f"Number of properties in group '{value.name}' "
                    + "differ from the number of 'channels'."
                )

            if value.name not in self.metadata["EM Dataset"]["Property groups"]:
                self.metadata["EM Dataset"]["Property groups"].append(value.name)

    def _fetch_transmitter_id(self) -> ReferencedData | IntegerData | None:
        """
        Utility method to retrieve the transmitter ID property, either from
        metadata or from list of children Data.
        """
        if self.type == "Receivers":
            uid = self.metadata["EM Dataset"].get("Tx ID property", None)
        else:
            uid = self.metadata["EM Dataset"].get("Tx ID tx property", None)

        if uid is not None:
            tx_id_property = self.get_entity(uid)[0]
        else:
            tx_id_property = self.get_entity("Transmitter ID")[0]
            # Reset the metadata

        if not isinstance(tx_id_property, ReferencedData | IntegerData | type(None)):
            raise TypeError(
                "Transmitter ID property must be of type ReferencedData or IntegerData."
            )

        if uid is None and self.type == "Receivers":
            self.edit_em_metadata(
                {"Tx ID property": getattr(tx_id_property, "uid", None)}
            )
        elif uid is None:
            self.edit_em_metadata(
                {"Tx ID tx property": getattr(tx_id_property, "uid", None)}
            )

        return tx_id_property

    def validate_em_metadata(self, values: dict | np.ndarray | bytes | None) -> dict:
        """
        Validate and format the metadata structure for EM entities.

        :param values: Metadata dictionary.

        :return: Validated and formatted metadata dictionary.
        """
        if values is None:
            metadata = self.default_metadata

            if self.type is not None:
                metadata["EM Dataset"][self.type] = self.uid

            values = metadata

        if isinstance(values, np.ndarray):
            values = values[0]

        if isinstance(values, bytes):
            values = str_json_to_dict(values)

        if not isinstance(values, dict):
            raise TypeError("'metadata' must be of type 'dict'")

        if "EM Dataset" not in values:
            values = {"EM Dataset": values}

        missing_keys = []
        for key in self.default_metadata["EM Dataset"]:
            if key not in values["EM Dataset"]:
                missing_keys += [key]

        if missing_keys:
            raise KeyError(
                f"'{missing_keys}' argument(s) missing from the input metadata."
            )

        for key, value in values["EM Dataset"].items():
            values["EM Dataset"][key] = str2uuid(value)

        if "Property groups" in values["EM Dataset"]:
            prop_groups = []
            for value in values["EM Dataset"]["Property groups"]:
                prop_group = self.get_property_group(str2uuid(value))[0]

                if isinstance(prop_group, PropertyGroup):
                    prop_groups.append(prop_group.name)

            values["EM Dataset"]["Property groups"] = prop_groups

        return values

    def _super_copy(
        self,
        parent: Group | Workspace | None = None,
        copy_children: bool = True,
        clear_cache: bool = False,
        mask: np.ndarray | None = None,
        **kwargs,
    ):
        """
        Call the super().copy of the class in copy_complement method.

        :return: New copy of the input entity.
        """
        return super().copy(
            parent=parent,
            copy_children=copy_children,
            clear_cache=clear_cache,
            mask=mask,
            **kwargs,
        )

    @property
    def tx_id_property(self) -> ReferencedData | IntegerData | None:
        """
        Data link between the receiver and transmitter object.
        """
        if self._tx_id_property is None and self.metadata is not None:
            self._tx_id_property = self._fetch_transmitter_id()

        return self._tx_id_property

    @tx_id_property.setter
    def tx_id_property(self, value: uuid.UUID | ReferencedData | np.ndarray | None):
        if isinstance(value, uuid.UUID):
            value = self.get_data(value)[0]

        if isinstance(value, np.ndarray):
            attributes = {
                "values": value.astype(np.int32),
            }
            self._format_transmitter_ids(value, attributes)
            value = self.add_data({"Transmitter ID": attributes})

        if not isinstance(value, (ReferencedData, IntegerData, type(None))):
            raise TypeError(
                "Input value for 'tx_id_property' should be of type uuid.UUID, "
                "ReferencedData, np.ndarray or None.)"
            )

        self._tx_id_property = value

        if self.type == "Receivers":
            self.edit_em_metadata({"Tx ID property": getattr(value, "uid", None)})
        else:
            self.edit_em_metadata({"Tx ID tx property": getattr(value, "uid", None)})

    def _format_transmitter_ids(self, _, attributes):
        if self.complement is not None and self.complement.tx_id_property is not None:
            attributes["entity_type"] = self.complement.tx_id_property.entity_type


class MovingLoopGroundEMSurvey(BaseEMSurvey, Curve, ABC):
    __INPUT_TYPE = ["Rx"]
    _TYPE_UID: uuid.UUID | None = None

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
        self.edit_em_metadata({"Loop radius": value})


class LargeLoopGroundEMSurvey(BaseEMSurvey, Curve, ABC):
    __INPUT_TYPE = ["Tx and Rx"]
    _TYPE_UID: uuid.UUID | None = None

    @property
    def base_receiver_type(self):
        return Curve

    @property
    def base_transmitter_type(self):
        return Curve

    def copy_complement(
        self,
        new_entity,
        *,
        parent: Group | Workspace | None = None,
        copy_children: bool = True,
        clear_cache: bool = False,
        mask: np.ndarray | None = None,
    ):
        new_complement = super().copy_complement(
            new_entity,
            parent=parent,
            copy_children=copy_children,
            clear_cache=clear_cache,
            mask=mask,
        )

        # Re-number the value_map for tx_id_property to remain
        if new_complement is not None and isinstance(
            new_complement.tx_id_property, ReferencedData
        ):
            value_map = {
                val: ind
                for ind, val in enumerate(
                    np.r_[0, np.unique(new_entity.transmitters.tx_id_property.values)]
                )
            }
            new_map = new_complement.tx_id_property.entity_type.validate_value_map(
                {
                    val: dict(new_entity.transmitters.tx_id_property.value_map.map)[ind]
                    for ind, val in value_map.items()
                }
            )
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

    def _format_transmitter_ids(self, values, attributes):
        if self.complement is not None and self.complement.tx_id_property is not None:
            attributes["entity_type"] = self.complement.tx_id_property.entity_type
        else:
            value_map = {
                ind: f"Loop {ind}" for ind in np.unique(values.astype(np.int32))
            }
            value_map[0] = "Unknown"
            attributes.update(
                {
                    "primitive_type": "REFERENCED",
                    "value_map": value_map,
                    "association": "VERTEX",
                }
            )


class AirborneEMSurvey(BaseEMSurvey, Curve, ABC):
    __INPUT_TYPE = ["Rx", "Tx", "Tx and Rx"]
    _PROPERTY_MAP = {
        "crossline_offset": "Crossline offset",
        "inline_offset": "Inline offset",
        "pitch": "Pitch",
        "roll": "Roll",
        "vertical_offset": "Vertical offset",
        "yaw": "Yaw",
    }
    _TYPE_UID: uuid.UUID | None = None

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
            self.edit_em_metadata({field + " value": value, field + " property": None})
        elif isinstance(value, uuid.UUID):
            self.edit_em_metadata({field + " value": None, field + " property": value})
        elif value is None:
            self.edit_em_metadata({field + " value": None, field + " property": None})
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
        self.edit_em_metadata({"Loop radius": value})

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
        self.edit_em_metadata({"Angles relative to bearing": value})

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


class FEMSurvey(BaseEMSurvey, ABC):
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


class TEMSurvey(BaseEMSurvey, ABC):
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

        self.edit_em_metadata({"Waveform": value})

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

        self.edit_em_metadata({"Waveform": value})

    @property
    def waveform_parameters(self) -> dict | None:
        """Access the waveform parameters stored as a dictionary."""
        waveform = self.get_data("_waveform_parameters")[0]

        if waveform is not None:
            return json.loads(waveform.values)

        return None
