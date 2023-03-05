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

from geoh5py.objects.object_base import ObjectType
from geoh5py.objects.points import Points

from .base import BaseEMSurvey


class MTReceivers(BaseEMSurvey, Points):
    """
    A magnetotellurics survey object.
    """

    __TYPE_UID = uuid.UUID("{b99bd6e5-4fe1-45a5-bd2f-75fc31f91b38}")
    __TYPE = "Receivers"
    __UNITS = [
        "Hertz (Hz)",
        "KiloHertz (kHz)",
        "MegaHertz (MHz)",
        "Gigahertz (GHz)",
    ]
    __INPUT_TYPE = ["Rx only"]

    def __init__(self, object_type: ObjectType, name="Magnetotellurics rx", **kwargs):
        super().__init__(object_type, name=name, **kwargs)

    def copy(
        self,
        parent=None,
        copy_children: bool = True,
        clear_cache: bool = False,
        mask: np.ndarray | None = None,
        **kwargs,
    ):
        """
        Function to copy an entity to a different parent entity.

        :param parent: Target parent to copy the entity under. Copied to current
            :obj:`~geoh5py.shared.entity.Entity.parent` if None.
        :param copy_children: (Optional) Create copies of all children entities along with it.
        :param clear_cache: Clear array attributes after copy.
        :param mask: Array of indices to sub-sample the input entity.
        :param kwargs: Additional keyword arguments.

        :return: New copy of the input entity.
        """
        if parent is None:
            parent = self.parent

        if mask is not None and self.vertices is not None:
            if not isinstance(mask, np.ndarray) or mask.shape != (
                self.vertices.shape[0],
            ):
                raise ValueError("Mask must be an array of shape (n_vertices,).")

            kwargs.update({"vertices": self.vertices[mask]})

        new_entity = parent.workspace.copy_to_parent(
            self,
            parent,
            clear_cache=clear_cache,
            **kwargs,
        )

        if copy_children:
            children_map = {}
            for child in self.children:
                child_copy = child.copy(
                    parent=new_entity, copy_children=True, mask=mask
                )
                children_map[child.uid] = child_copy.uid

            if self.property_groups:
                self.workspace.copy_property_groups(
                    new_entity, self.property_groups, children_map
                )
                new_entity.workspace.update_attribute(new_entity, "property_groups")

        return new_entity

    @property
    def default_input_types(self) -> list[str]:
        """Choice of survey creation types."""
        return self.__INPUT_TYPE

    @property
    def default_metadata(self) -> dict:
        """
        :return: Default unique identifier
        """
        return {
            "EM Dataset": {
                "Channels": [],
                "Input type": "Rx only",
                "Property groups": [],
                "Receivers": None,
                "Survey type": "Magnetotellurics",
                "Unit": "Hertz (Hz)",
            }
        }

    @property
    def default_receiver_type(self):
        """
        :return: Transmitter class
        """
        return MTReceivers

    @property
    def default_transmitter_type(self):
        """
        :return: Transmitter class
        """
        return type(None)

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

    @property
    def type(self):
        """Survey element type"""
        return self.__TYPE
