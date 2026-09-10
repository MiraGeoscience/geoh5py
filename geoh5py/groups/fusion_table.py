# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2020-2026 Mira Geoscience Ltd.                                '
#                                                                              '
#  This file is part of geoh5py.                                               '
#                                                                              '
#  geoh5py is free software: you can redistribute it and/or modify             '
#  it under the terms of the GNU Lesser General Public License as published by '
#  the Free Software Foundation, either version 3 of the License, or           '
#  (at your option) any later version.                                         '
#                                                                              '
#  geoh5py is distributed in the hope that it will be useful,                  '
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              '
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               '
#  GNU Lesser General Public License for more details.                         '
#                                                                              '
#  You should have received a copy of the GNU Lesser General Public License    '
#  along with geoh5py.  If not, see <https://www.gnu.org/licenses/>.           '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


from __future__ import annotations

import uuid
from numbers import Integral

import numpy as np

from ..data import NumericData, ReferencedData
from ..shared.entity import Entity
from ..shared.utils import str2uuid
from .base import Group


class FusionTableGroup(Group):
    """Group for Data Fusion Table."""

    _TYPE_UID = uuid.UUID("{d9576a5e-07e9-42ee-8b83-6b4b1cd992d4}")
    _default_name = "Data Fusion Table"

    _attribute_map = Entity._attribute_map.copy()  # pylint: disable=protected-access
    _attribute_map.update(
        {
            "Can add group": "can_add_group",
            "mesh": "mesh",
            "negativeIndex": "negative_index",
            "positiveIndex": "positive_index",
            "referenceData": "reference_data",
            "testIndex": "test_index",
        }
    )

    def __init__(
        self,
        *,
        mesh: uuid.UUID | None = None,
        can_add_group: bool = True,
        feature_list: list[dict] | None = None,
        negative_index: int = 1,
        positive_index: int = 2,
        reference_data: uuid.UUID | None = None,
        test_index: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.can_add_group = can_add_group
        self.mesh = mesh
        self.reference_data = reference_data
        self._feature_list = self._validate_feature_list(feature_list)
        self.negative_index = negative_index
        self.positive_index = positive_index
        self.test_index = test_index

    @property
    def can_add_group(self) -> bool:
        """
        Boolean indicating whether the fusion table can add a group.
        """
        return self._can_add_group

    @can_add_group.setter
    def can_add_group(self, value: bool):
        if not isinstance(value, bool | np.integer) or value not in (0, 1):
            raise TypeError("Attribute 'can_add_group' must be a boolean value.")
        self._can_add_group = bool(value)

        if self.on_file:
            self.workspace.update_attribute(self, "can_add_group")

    @property
    def feature_list(self) -> list[dict] | None:
        """
        Dictionary containing the features of the fusion table.
        """
        return self._feature_list

    def _validate_feature_list(self, value: list[dict] | None):
        if value is None and self.mesh is not None:
            value = []
            mesh_entity = self.workspace.get_entity(self.mesh)[0]

            if mesh_entity is not None and hasattr(mesh_entity, "children"):
                for child in mesh_entity.children:
                    if (
                        isinstance(child, NumericData)
                        and child.uid != self.reference_data
                    ):
                        value.append({"id": child.uid, "isChecked": True, "notes": ""})

        if not isinstance(value, list | None):
            raise TypeError("Attribute 'feature_list' must be a list or None.")

        return value

    @property
    def mesh(self) -> uuid.UUID | None:
        """
        ID of the object containing the data for the fusion table.
        """
        return self._mesh

    @mesh.setter
    def mesh(self, value: uuid.UUID | Entity | str | None):

        if isinstance(value, Entity):
            value = value.uid
        elif isinstance(value, str | uuid.UUID):
            value = str2uuid(value)

        if not isinstance(value, uuid.UUID | None):
            raise TypeError("Attribute 'mesh' must be a UUID or Entity.")

        if self.on_file and value is not None:
            mesh_entity = self.workspace.get_entity(value)[0]

            if mesh_entity is None:
                raise ValueError("The 'mesh' must be a valid entity in the workspace.")

            if self.get_entity(value)[0] is None and hasattr(mesh_entity, "copy"):
                new_mesh = mesh_entity.copy(parent=self)
                value = new_mesh.uid

        if not isinstance(value, uuid.UUID | None):
            raise TypeError("Attribute 'mesh' must be a UUID or Entity.")

        self._mesh = value

        if self.on_file:
            self.workspace.update_attribute(self, "attributes")

    @property
    def negative_index(self) -> np.uint32 | None:
        """
        Index of the negative data in the fusion table.
        """
        return self._negative_index

    @negative_index.setter
    def negative_index(self, value: int | None):
        if not isinstance(value, Integral | None):
            raise TypeError("Attribute 'negative_index' must be an integer or None.")

        self._negative_index = np.uint32(value)

        if self.on_file:
            self.workspace.update_attribute(self, "attributes")

    @property
    def positive_index(self) -> np.uint32 | None:
        """
        Index of the positive data in the fusion table.
        """
        return self._positive_index

    @positive_index.setter
    def positive_index(self, value: int | None):
        if not isinstance(value, Integral | None):
            raise TypeError("Attribute 'positive_index' must be an integer or None.")

        self._positive_index = np.uint32(value)

        if self.on_file:
            self.workspace.update_attribute(self, "attributes")

    @property
    def reference_data(self) -> uuid.UUID | None:
        """
        ID of the reference data containing the positive, negative and test indices.
        """
        return self._reference_data

    @reference_data.setter
    def reference_data(self, value: uuid.UUID | ReferencedData | None):
        value = str2uuid(value)

        if isinstance(value, ReferencedData):
            value = value.uid

        if not isinstance(value, uuid.UUID | None):
            raise TypeError("Attribute 'reference_data' must be a UUID or None.")

        if self._mesh is not None and value is not None:
            mesh_entity = self.workspace.get_entity(self._mesh)[0]

            if mesh_entity is not None and hasattr(mesh_entity, "get_data"):
                data = mesh_entity.get_data(value)
                if not data or not isinstance(data[0], ReferencedData):
                    raise ValueError(
                        "The 'reference_data' must be a valid ReferenceData entity in the mesh."
                    )

        self._reference_data = value

        if self.on_file:
            self.workspace.update_attribute(self, "attributes")

    def set_active(self, active: bool, data: NumericData | uuid.UUID):
        """
        Set a data in the fusion table as active or inactive.

        :param active: Boolean indicating whether to set the data as active or inactive.
        :param data: The data object or UUID to set as active or inactive.
        """
        if isinstance(data, NumericData):
            data = data.uid
        elif not isinstance(data, uuid.UUID):
            raise TypeError("The 'data' must be a NumericData object or a UUID.")

        if self.feature_list is None:
            raise ValueError("The 'feature_list' is not set. Cannot set active state.")

        for elem in self.feature_list:
            if elem["id"] == data:
                elem["isChecked"] = bool(active)
                break

        self.workspace.update_attribute(self, "feature_list")

    @property
    def test_index(self) -> np.uint32 | None:
        """
        Index of the test data in the fusion table.
        """
        return self._test_index

    @test_index.setter
    def test_index(self, value: int | None):
        if not isinstance(value, Integral | None):
            raise TypeError("Attribute 'test_index' must be an integer or None.")

        self._test_index = np.uint32(value)

        if self.on_file:
            self.workspace.update_attribute(self, "attributes")
