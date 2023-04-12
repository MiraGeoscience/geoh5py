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

# pylint: disable=bad-super-call

from __future__ import annotations

import uuid

import numpy as np

from geoh5py.objects.curve import Curve
from geoh5py.objects.object_type import ObjectType
from geoh5py.objects.points import Points

from .base import FEMSurvey, AirborneEMSurvey


class TipperSurvey(FEMSurvey, AirborneEMSurvey):  # pylint: disable=too-many-ancestors
    """
    Base tipper survey class.
    """


    __INPUT_TYPE = ["Rx and base stations"]
    _base_stations = None
    _receivers = None

    def __init__(
        self,
        object_type: ObjectType,
        base_stations: TipperBaseStations | None = None,
        **kwargs,
    ):
        self._base_stations = base_stations

        super().__init__(
            object_type,
            **kwargs,
        )

    @property
    def base_stations(self) -> TipperBaseStations | None:
        """The base station entity"""
        if isinstance(self, TipperBaseStations):
            return self

        if getattr(self, "_base_stations", None) is None:
            if (
                self.metadata is not None
                and "Base stations" in self.metadata["EM Dataset"]
            ):
                base_station = self.metadata["EM Dataset"]["Base stations"]
                base_station_entity = self.workspace.get_entity(base_station)[0]

                if isinstance(base_station_entity, TipperBaseStations):
                    self._base_stations = base_station_entity

        return self._base_stations

    @base_stations.setter
    def base_stations(self, base: TipperBaseStations):
        if not isinstance(base, (TipperBaseStations, type(None))):
            raise TypeError(
                f"Input `base_stations` must be of type '{TipperBaseStations}' or None"
            )

        if isinstance(self, TipperBaseStations):
            raise AttributeError(
                f"The 'base_station' attribute cannot be set on class {TipperBaseStations}."
            )

        if base.n_vertices not in [self.n_vertices, 1, None]:
            raise AttributeError(
                "The input 'base_stations' should have n_vertices equal to 1, n_receivers or None."
            )

        self._base_stations = base
        self.edit_metadata({"Base stations": base.uid})

    def copy_from_extent(
        self,
        extent: np.ndarray,
        parent=None,
        copy_children: bool = True,
        clear_cache: bool = False,
        inverse: bool = False,
        **kwargs,
    ) -> TipperReceivers | TipperBaseStations | None:
        """
        Sub-class extension of :func:`~geoh5py.shared.entity.Entity.copy_from_extent`.
        """
        indices = self.mask_by_extent(extent, inverse=inverse)

        if indices is None:
            return None

        new_entity = self.copy(
            parent=parent,
            copy_children=copy_children,
            clear_cache=clear_cache,
            mask=indices,
            **kwargs,
        )

        return new_entity

    @property
    def default_input_types(self) -> list[str]:
        """Choice of survey creation types."""
        return self.__INPUT_TYPE

    @property
    def default_receiver_type(self):
        """
        :return: Transmitter class
        """
        return TipperReceivers

    @property
    def default_transmitter_type(self):
        """
        :return: Transmitter class
        """
        return type(None)

    @property
    def base_receiver_type(self):
        return Curve

    @property
    def base_transmitter_type(self):
        return Points

    @property
    def default_metadata(self) -> dict:
        """
        :return: Default unique identifier
        """
        return {
            "EM Dataset": {
                "Base stations": None,
                "Channels": [],
                "Input type": "Rx and base stations",
                "Property groups": [],
                "Receivers": None,
                "Survey type": "ZTEM",
                "Unit": "Hertz (Hz)",
            }
        }

    @property
    def default_units(self) -> list[str]:
        """Accepted time units. Must be one of "Seconds (s)",
        "Milliseconds (ms)", "Microseconds (us)" or "Nanoseconds (ns)"
        """
        return self.__UNITS


class TipperReceivers(TipperSurvey, Curve):  # pylint: disable=too-many-ancestors
    """
    A z-tipper EM survey object.
    """

    __TYPE_UID = uuid.UUID("{0b639533-f35b-44d8-92a8-f70ecff3fd26}")
    __TYPE = "Receivers"

    def __init__(self, object_type: ObjectType, name="Tipper rx", **kwargs):
        self._base_stations: TipperBaseStations | None = None

        super().__init__(object_type, name=name, **kwargs)

    @property
    def complement(self):
        return self.base_stations

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


class TipperBaseStations(TipperSurvey, Points):
    """
    A z-tipper EM survey object.
    """

    __TYPE_UID = uuid.UUID("{f495cd13-f09b-4a97-9212-2ea392aeb375}")
    __TYPE = "Base stations"

    def __init__(self, object_type: ObjectType, name="Tipper base", **kwargs):
        super().__init__(object_type, name=name, **kwargs)

    @property
    def complement(self):
        return self.receivers

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
