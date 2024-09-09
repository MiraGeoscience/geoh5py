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

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np

from ..shared import Entity
from ..shared.utils import mask_by_extent
from .data_association_enum import DataAssociationEnum
from .data_type import DataType, ReferenceDataType
from .primitive_type_enum import PrimitiveTypeEnum


class Data(Entity):
    """
    Base class for Data entities.
    """

    _attribute_map = Entity._attribute_map.copy()
    _attribute_map.update({"Association": "association", "Modifiable": "modifiable"})
    _visible = False

    def __init__(
        self,
        association: DataAssociationEnum = DataAssociationEnum.OBJECT,
        modifiable: bool = True,
        visible: bool = False,
        values: Any | None = None,
        **kwargs,
    ):
        self._association = self.validate_association(association)

        super().__init__(visible=visible, **kwargs)

        self.modifiable = modifiable
        self.values = values

    def copy(
        self,
        parent=None,
        clear_cache: bool = False,
        mask: np.ndarray | None = None,
        **kwargs,
    ) -> Data:
        """
        Function to copy data to a different parent entity.

        :param parent: Target parent to copy the entity under. Copied to current
            :obj:`~geoh5py.shared.entity.Entity.parent` if None.
        :param clear_cache: Clear array attributes after copy.
        :param mask: Array of bool defining the values to keep.
        :param kwargs: Additional keyword arguments to pass to the copy constructor.

        :return entity: Registered Entity to the workspace.
        """
        if parent is None:
            parent = self.parent

        if self.values is not None and mask is not None:
            if not isinstance(mask, np.ndarray):
                raise TypeError("Mask must be an array or None.")

            if mask.dtype != bool or mask.shape != self.values.shape:
                raise ValueError(
                    f"Mask must be a boolean array of shape {self.values.shape}, not {mask.shape}"
                )

            n_values = (
                parent.n_cells
                if self.association is DataAssociationEnum.CELL
                else parent.n_vertices
            )

            if n_values < self.values.shape[0]:
                kwargs.update({"values": self.values[mask]})
            else:
                values = np.ones_like(self.values) * self.nan_value
                values[mask] = self.values[mask]

                kwargs.update({"values": values})

        new_entity = parent.workspace.copy_to_parent(
            self,
            parent,
            clear_cache=clear_cache,
            **kwargs,
        )

        return new_entity

    def copy_from_extent(
        self,
        extent: np.ndarray,
        parent=None,
        clear_cache: bool = False,
        inverse: bool = False,
        **kwargs,
    ) -> Entity | None:
        """
        Function to copy data based on a bounding box extent.

        :param extent: Bounding box extent requested for the input entity, as supplied for
            :func:`~geoh5py.shared.entity.Entity.mask_by_extent`.
        :param parent: Target parent to copy the entity under. Copied to current
            :obj:`~geoh5py.shared.entity.Entity.parent` if None.
        :param clear_cache: Clear array attributes after copy.
        :param inverse: Keep the inverse (clip) of the extent selection.
        :param kwargs: Additional keyword arguments to pass to the copy constructor.

        :return entity: Registered Entity to the workspace.
        """
        indices = self.mask_by_extent(extent, inverse=inverse)
        if indices is None:
            return None

        return self.copy(
            parent=parent,
            clear_cache=clear_cache,
            mask=indices,
            **kwargs,
        )

    @property
    def n_values(self) -> int | None:
        """
        :obj:`int`: Number of expected data values based on
        :obj:`~geoh5py.data.data.Data.association`
        """
        if self.association in [
            DataAssociationEnum.VERTEX,
            DataAssociationEnum.DEPTH,
        ] and hasattr(self.parent, "n_vertices"):
            return self.parent.n_vertices
        if self.association is DataAssociationEnum.CELL and hasattr(
            self.parent, "n_cells"
        ):
            return self.parent.n_cells
        if self.association is DataAssociationEnum.FACE and hasattr(
            self.parent, "n_faces"
        ):
            return self.parent.n_faces
        if self.association is DataAssociationEnum.OBJECT:
            return 1

        return None

    @property
    def nan_value(self) -> None:
        """
        Value used to represent missing data in python.
        """
        return None

    @property
    def association(self) -> DataAssociationEnum:
        """
        :obj:`~geoh5py.data.data_association_enum.DataAssociationEnum`:
        Relationship made between the
        :func:`~geoh5py.data.data.Data.values` and elements of the
        :obj:`~geoh5py.shared.entity.Entity.parent` object.
        Association can be set from a :obj:`str` chosen from the list of available
        :obj:`~geoh5py.data.data_association_enum.DataAssociationEnum` options.
        """
        return self._association

    @property
    def entity_type(self) -> DataType | ReferenceDataType:
        """
        Type of data.
        """
        return self._entity_type

    @entity_type.setter
    def entity_type(self, data_type: DataType | ReferenceDataType):
        self._entity_type = data_type

        if self.on_file:
            self.workspace.update_attribute(self, "entity_type")

    @property
    def modifiable(self) -> bool:
        """
        :obj:`bool` Entity can be modified.
        """
        return self._modifiable

    @modifiable.setter
    def modifiable(self, value: bool):
        self._modifiable = value

        if self.on_file:
            self.workspace.update_attribute(self, "attributes")

    @classmethod
    @abstractmethod
    def primitive_type(cls) -> PrimitiveTypeEnum:
        """Abstract method to return the primitive type of the class."""

    def mask_by_extent(
        self, extent: np.ndarray, inverse: bool = False
    ) -> np.ndarray | None:
        """
        Sub-class extension of :func:`~geoh5py.shared.entity.Entity.mask_by_extent`.

        Uses the parent object's vertices or centroids coordinates.
        """
        if self.association is DataAssociationEnum.VERTEX and hasattr(
            self.parent, "vertices"
        ):
            return mask_by_extent(self.parent.vertices, extent, inverse=inverse)

        if self.association is DataAssociationEnum.CELL:
            if hasattr(self.parent, "centroids"):
                return mask_by_extent(self.parent.centroids, extent, inverse=inverse)

            if hasattr(self.parent, "vertices") and hasattr(self.parent, "cells"):

                indices = mask_by_extent(self.parent.vertices, extent, inverse=inverse)
                if indices is not None:
                    indices = np.all(indices[self.parent.cells], axis=1)

                return indices

        return None

    def validate_entity_type(self, entity_type: DataType | None) -> DataType:
        """
        Validate the entity type.
        """
        if (
            not isinstance(entity_type, DataType)
            or entity_type.primitive_type != self.primitive_type()
        ):
            raise TypeError(
                "Input 'entity_type' must be a DataType object of primitive_type 'TEXT'."
            )

        if entity_type.name == "Entity":
            entity_type.name = self.name

        return entity_type

    @staticmethod
    def validate_association(value: str | DataAssociationEnum):
        if isinstance(value, str):
            if value.upper() not in DataAssociationEnum.__members__:
                raise ValueError(
                    f"Association flag should be one of {DataAssociationEnum.__members__}"
                )

            value = getattr(DataAssociationEnum, value.upper())

        if not isinstance(value, DataAssociationEnum):
            raise TypeError(f"Association must be of type {DataAssociationEnum}")

        return value

    @abstractmethod
    def validate_values(self, values: Any | None) -> Any:
        """
        Validate the values.
        """

    @property
    def values(self):
        """
        Data values
        """
        if getattr(self, "_values", None) is None:
            values = self.workspace.fetch_values(self)
            self._values = self.validate_values(values)

        return self._values

    @values.setter
    def values(self, values: np.ndarray | None):
        self._values = self.validate_values(values)

        if self.on_file:
            self.workspace.update_attribute(self, "values")

    def __call__(self):
        return self.values
