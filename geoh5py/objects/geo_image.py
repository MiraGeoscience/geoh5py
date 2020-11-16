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
from typing import List, Optional, Union

import numpy as np

from ..data import Data
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

    def add_data(
        self, data: dict, property_group: str = None
    ) -> Union[Data, List[Data]]:
        """
        Create :obj:`~geoh5py.data.data.Data` from dictionary of name and arguments.
        The provided arguments can be any property of the target Data class.

        :param data: Dictionary of data to be added to the object, e.g.

        .. code-block:: python

            data = {
                "data_A": {
                    'values', [v_1, v_2, ...],
                    'association': 'VERTEX'
                    },
                "data_B": {
                    'values', [v_1, v_2, ...],
                    'association': 'CELLS'
                    },
            }

        :return: List of new Data objects.
        """
        data_objects = []
        for name, attr in data.items():
            assert isinstance(attr, dict), (
                f"Given value to data {name} should of type {dict}. "
                f"Type {type(attr)} given instead."
            )
            assert "values" in list(
                attr.keys()
            ), f"Given attr for data {name} should include 'values'"

            attr["name"] = name

            if "association" not in list(attr.keys()):
                if (
                    getattr(self, "n_cells", None) is not None
                    and attr["values"].ravel().shape[0] == self.n_cells
                ):
                    attr["association"] = "CELL"
                elif (
                    getattr(self, "n_vertices", None) is not None
                    and attr["values"].ravel().shape[0] == self.n_vertices
                ):
                    attr["association"] = "VERTEX"
                else:
                    attr["association"] = "OBJECT"

            if "entity_type" in list(attr.keys()):
                entity_type = attr["entity_type"]
            else:
                if isinstance(attr["values"], np.ndarray):
                    entity_type = {"primitive_type": "FLOAT"}
                elif isinstance(attr["values"], str):
                    entity_type = {"primitive_type": "TEXT"}
                else:
                    raise NotImplementedError(
                        "Only add_data values of type FLOAT and TEXT have been implemented"
                    )

            # Re-order to set parent first
            kwargs = {"parent": self, "association": attr["association"]}
            for key, val in attr.items():
                if key in ["parent", "association", "entity_type"]:
                    continue
                kwargs[key] = val

            data_object = self.workspace.create_entity(
                Data, entity=kwargs, entity_type=entity_type
            )

            if property_group is not None:
                self.add_data_to_group(data_object, property_group)

            data_objects.append(data_object)

        if len(data_objects) == 1:
            return data_object

        return data_objects
