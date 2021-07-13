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
from typing import Optional

import numpy as np

from .object_base import ObjectBase, ObjectType


class Points(ObjectBase):
    """
    Points object made up of vertices.
    """

    __TYPE_UID = uuid.UUID("{202C5DB1-A56D-4004-9CAD-BAAFD8899406}")

    def __init__(self, object_type: ObjectType, **kwargs):
        self._vertices: np.ndarray = None

        super().__init__(object_type, **kwargs)

        object_type.workspace._register_object(self)

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID

    @property
    def vertices(self) -> Optional[np.ndarray]:
        """
        :obj:`~geoh5py.objects.object_base.ObjectBase.vertices`
        """
        if (getattr(self, "_vertices", None) is None) and self.existing_h5_entity:
            self._vertices = self.workspace.fetch_coordinates(self.uid, "vertices")

        if getattr(self, "_vertices", None) is not None:
            return self._vertices.view("<f8").reshape((-1, 3))

        return None

    @vertices.setter
    def vertices(self, xyz: np.ndarray):
        self.modified_attributes = "vertices"
        assert (
            xyz.shape[1] == 3
        ), f"Array of vertices must be of shape (*, 3). Array of shape {xyz.shape} provided."
        self._vertices = np.asarray(
            np.core.records.fromarrays(
                xyz.T.tolist(),
                dtype=[("x", "<f8"), ("y", "<f8"), ("z", "<f8")],
            )
        )
