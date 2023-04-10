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

from .base import BaseAirborneEMSurvey


class BaseAirborneFEMSurvey(BaseAirborneEMSurvey, Curve):  # pylint: disable=too-many-ancestors


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

        complement: AirborneFEMTransmitters | AirborneFEMReceivers = (
            self.transmitters  # type: ignore
            if isinstance(self, AirborneFEMReceivers)
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
                "Survey type": "Airborne FEM",
                "Transmitters": None,
                "Unit": "Hertz (Hz)",
            }
        }

    @property
    def default_receiver_type(self):
        """
        :return: Transmitter class
        """
        return AirborneFEMReceivers

    @property
    def default_transmitter_type(self):
        """
        :return: Transmitter class
        """
        return AirborneFEMTransmitters




class AirborneFEMReceivers(BaseAirborneFEMSurvey):  # pylint: disable=too-many-ancestors
    """
    Airborne frequency-domain electromagnetic receivers class.
    """

    __TYPE_UID = uuid.UUID("{19730589-fd28-4649-9de0-ad47249d9aba}")
    __TYPE = "Receivers"

    def __init__(self, object_type: ObjectType, name="Airborne FEM Rx", **kwargs):
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


class AirborneFEMTransmitters(BaseAirborneFEMSurvey):  # pylint: disable=too-many-ancestors
    """
    Airborne frequency-domain electromagnetic transmitters class.
    """

    __TYPE_UID = uuid.UUID("{58c4849f-41e2-4e09-b69b-01cf4286cded}")
    __TYPE = "Transmitters"

    def __init__(self, object_type: ObjectType, name="Airborne FEM Tx", **kwargs):
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
