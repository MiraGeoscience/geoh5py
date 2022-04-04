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

import numpy as np

from .object_base import ObjectBase, ObjectType


class DrapeModel(ObjectBase):
    """
    Drape (curtain) model object made up of layers and prisms.
    """

    __TYPE_UID = uuid.UUID("{C94968EA-CF7D-11EB-B8BC-0242AC130003}")

    def __init__(self, object_type: ObjectType, **kwargs):
        self._layers: np.ndarray | None = None
        self._prisms: np.ndarray | None = None

        super().__init__(object_type, **kwargs)

        object_type.workspace._register_object(self)

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID

    @property
    def layers(self) -> np.ndarray | None:
        """
        :obj:`~geoh5py.objects.object_base.ObjectBase.layers`
        """
        if self._layers is None and self.existing_h5_entity:
            self._layers = self.workspace.fetch_coordinates(self.uid, "layers")

        if self._layers is not None:
            return np.array(self._layers.tolist())

        return None

    @layers.setter
    def layers(self, xyz: np.ndarray):
        self.modified_attributes = "layers"
        assert (
            xyz.shape[1] == 3
        ), f"Array of layers must be of shape (*, 3). Array of shape {xyz.shape} provided."
        self._layers = np.asarray(
            np.core.records.fromarrays(
                xyz.T.tolist(),
                dtype=[("I", "<i4"), ("K", "<i4"), ("Bottom elevation", "<f8")],
            )
        )

    @property
    def prisms(self) -> np.ndarray | None:
        """
        :obj:`~geoh5py.objects.object_base.ObjectBase.prisms`
        """
        if self._prisms is None and self.existing_h5_entity:
            self._prisms = self.workspace.fetch_coordinates(self.uid, "prisms")

        if self._prisms is not None:
            return np.array(self._prisms.tolist())

        return None

    @prisms.setter
    def prisms(self, xyz: np.ndarray):
        self.modified_attributes = "prisms"
        assert (
            xyz.shape[1] == 3
        ), f"Array of prisms must be of shape (*, 3). Array of shape {xyz.shape} provided."
        self._prisms = np.asarray(
            np.core.records.fromarrays(
                xyz.T.tolist(),
                dtype={
                    "names": ["Top elevation", "First layer", "Layer count"],
                    "formats": ["<f8", "<i4", "<i4"],
                    "offsets": [0, 24, 28],
                    "itemsize": 32,
                },
            )
        )
