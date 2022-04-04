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

from geoh5py.objects.object_type import ObjectType

from .base import BaseEMSurvey


class TipperReceivers(BaseEMSurvey):
    """
    A z-tipper EM survey object.
    """

    __TYPE_UID = uuid.UUID("{0b639533-f35b-44d8-92a8-f70ecff3fd26}")
    __METADATA = {
        "EM Dataset": {
            "Base stations": "",
            "Channels": [],
            "Input type": "Rx and base stations",
            "Property groups": [],
            "Receivers": "",
            "Survey type": "ZTEM",
            "Unit": "Hertz (Hz)",
        }
    }
    __UNITS = [
        "Hertz (Hz)",
        "KiloHertz (kHz)",
        "MegaHertz (MHz)",
        "Gigahertz (GHz)",
    ]
    __INPUT_TYPE = ["Rx and base stations"]

    def __init__(self, object_type: ObjectType, **kwargs):
        self._base_stations: TipperBaseStations | None = None

        super().__init__(object_type, **kwargs)

    @property
    def default_input_types(self) -> list[str]:
        """Input types. Must be 'Rx only'"""
        return self.__INPUT_TYPE

    @property
    def default_metadata(self) -> dict:
        """
        :return: Default unique identifier
        """
        return self.__METADATA

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        """
        :return: Default unique identifier
        """
        return cls.__TYPE_UID

    @property
    def default_units(self) -> list[str]:
        """Accepted time units. Must be one of "Seconds (s)",
        "Milliseconds (ms)", "Microseconds (us)" or "Nanoseconds (ns)"
        """
        return self.__UNITS

    def copy(self, parent=None, copy_children: bool = True) -> TipperReceivers:
        """
        Function to copy a TipperReceivers to a different parent entity.

        :param parent: Target parent to copy the entity under. Copied to current
            :obj:`~geoh5py.shared.entity.Entity.parent` if None.
        :param copy_children: Create copies of AirborneTEMReceivers along with it.

        :return entity: Registered TipperReceivers to the workspace.
        """
        if parent is None:
            parent = self.parent

        omit_list = ["_metadata", "_receivers", "_base_stations"]
        new_entity = parent.workspace.copy_to_parent(
            self, parent, copy_children=copy_children, omit_list=omit_list
        )
        new_base_stations = parent.workspace.copy_to_parent(
            self.base_stations,
            parent,
            copy_children=copy_children,
            omit_list=omit_list,
        )
        new_entity.base_stations = new_base_stations
        parent.workspace.finalize()

        return new_entity

    @property
    def receivers(self):
        """MT receivers"""
        return self

    @receivers.setter
    def receivers(self, value: BaseEMSurvey):
        raise AttributeError(
            "Attribute 'receivers' of the class 'MTReceivers' must reference to self. "
            f"Re-assignment to {value} ignored."
        )

    @property
    def base_stations(self):
        """The base station entity"""
        if getattr(self, "_base_stations", None) is None:
            if (
                self.metadata is not None
                and "Base stations" in self.metadata["EM Dataset"]
            ):
                tx_uid = self.metadata["EM Dataset"]["Base stations"]

                try:
                    self._base_stations = self.workspace.get_entity(tx_uid)[0]
                except IndexError:
                    print("Associated transmitters entity not found in Workspace.")
                    return None

        return self._base_stations

    @base_stations.setter
    def base_stations(self, base: TipperBaseStations):
        if not isinstance(base, (TipperBaseStations, type(None))):
            raise ValueError(
                f"Input `base_stations` must be of type '{TipperBaseStations}' or None"
            )

        self._base_stations = base
        self.edit_metadata({"Base stations": base.uid})


class TipperBaseStations(BaseEMSurvey):
    """
    A z-tipper EM survey object.
    """

    __TYPE_UID = uuid.UUID("{f495cd13-f09b-4a97-9212-2ea392aeb375}")

    def __init__(self, object_type: ObjectType, **kwargs):
        super().__init__(object_type, **kwargs)

    @property
    def default_receiver_type(self):
        """
        :return: Transmitter class
        """
        return TipperReceivers

    def copy(self, parent=None, copy_children: bool = True) -> TipperBaseStations:
        """
        Function to copy a TipperBaseStations to a different parent entity.

        :param parent: Target parent to copy the entity under. Copied to current
            :obj:`~geoh5py.shared.entity.Entity.parent` if None.
        :param copy_children: Create copies of AirborneTEMReceivers along with it.

        :return entity: Registered TipperBaseStations to the workspace.
        """
        if parent is None:
            parent = self.parent

        omit_list = ["_metadata", "_receivers", "_base_stations"]
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
