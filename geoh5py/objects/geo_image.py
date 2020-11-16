#  Copyright (c) 2020 Mira Geoscience Ltd.
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

import uuid
from typing import Optional

import numpy as np

from .object_base import ObjectBase, ObjectType


class GeoImage(ObjectBase):
    """
    Image object class.

    .. warning:: Not yet implemented.

    """

    __TYPE_UID = uuid.UUID(
        fields=(0x77AC043C, 0xFE8D, 0x4D14, 0x81, 0x67, 0x75E300FB835A)
    )

    def __init__(self, object_type: ObjectType, **kwargs):

        self._vertices = None
        self._cells = None

        super().__init__(object_type, **kwargs)

        if object_type.name == "None":
            self.entity_type.name = "GeoImage"

        object_type.workspace._register_object(self)

    @property
    def cells(self) -> Optional[np.ndarray]:
        r"""
        :obj:`numpy.ndarray` of :obj:`int`, shape (\*, 2):
        Array of indices defining segments connecting vertices. Defined based on
        :obj:`~geoh5py.objects.curve.Curve.parts` if set by the user.
        """
        if getattr(self, "_cells", None) is None:

            if self.existing_h5_entity:
                self._cells = self.workspace.fetch_cells(self.uid)
            else:
                if self.vertices is not None:
                    n_segments = self.vertices.shape[0]
                    self._cells = np.c_[
                        np.arange(0, n_segments - 1), np.arange(1, n_segments)
                    ].astype("uint32")

        return self._cells

    @cells.setter
    def cells(self, indices):
        assert indices.dtype == "uint32", "Indices array must be of type 'uint32'"
        self.modified_attributes = "cells"
        self._cells = indices

    @property
    def vertices(self) -> Optional[np.ndarray]:
        """
        :obj:`~geoh5py.objects.object_base.ObjectBase.vertices`:
        Defines the four corners of the geo_image
        """
        if (getattr(self, "_vertices", None) is None) and self.existing_h5_entity:
            self._vertices = self.workspace.fetch_vertices(self.uid)

        return self._vertices

    @vertices.setter
    def vertices(self, xyz: np.ndarray):
        self.modified_attributes = "vertices"
        self._vertices = xyz

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID
