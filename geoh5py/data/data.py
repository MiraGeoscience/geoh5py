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
from abc import abstractmethod

import numpy as np

from ..shared import Entity
from ..shared.utils import mask_by_extent
from .data_association_enum import DataAssociationEnum
from .data_type import DataType
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
        data_type: DataType,
        **kwargs,
    ):
        self._association = None
        self._on_file = False
        self._modifiable = True

        if (
            not isinstance(data_type, DataType)
            or data_type.primitive_type != self.primitive_type()
        ):
            raise TypeError(
                "Input 'data_type' must be a DataType object of primitive_type 'TEXT'."
            )

        self.entity_type = data_type
        self._values = None

        super().__init__(**kwargs)

        if self.entity_type.name == "Entity":
            self.entity_type.name = self.name

        data_type.workspace._register_data(self)

    def copy(
        self,
        parent=None,
        copy_children: bool = True,
        clear_cache: bool = False,
        mask: np.ndarray | None = None,
        **kwargs,
    ) -> Data:
        """
        Function to copy data to a different parent entity.

        :param parent: Target parent to copy the entity under. Copied to current
            :obj:`~geoh5py.shared.entity.Entity.parent` if None.
        :param copy_children: (Optional) Create copies of all children entities along with it.
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

            if mask.dtype != np.bool or mask.shape != self.values.shape:
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
                values = np.ones_like(self.values) * np.nan
                values[mask] = self.values[mask]
                kwargs.update({"values": values})

        new_entity = parent.workspace.copy_to_parent(
            self,
            parent,
            clear_cache=clear_cache,
            **kwargs,
        )

        return new_entity

    @property
    def extent(self) -> np.ndarray | None:
        """
        Geography bounding box of the parent object.

        :return: shape(2, 3) Bounding box defined by the bottom South-West and
            top North-East coordinates.
        """
        return self.parent.extent

    @property
    def n_values(self) -> int | None:
        """
        :obj:`int`: Number of expected data values based on
        :obj:`~geoh5py.data.data.Data.association`
        """
        if self.association is DataAssociationEnum.VERTEX:
            return self.parent.n_vertices
        if self.association is DataAssociationEnum.DEPTH:
            return self.parent.n_vertices
        if self.association is DataAssociationEnum.CELL:
            return self.parent.n_cells
        if self.association is DataAssociationEnum.FACE:
            return self.parent.n_faces
        if self.association is DataAssociationEnum.OBJECT:
            return 1

        return None

    @property
    def values(self):
        """
        Data values
        """
        return self._values

    @property
    def association(self) -> DataAssociationEnum | None:
        """
        :obj:`~geoh5py.data.data_association_enum.DataAssociationEnum`:
        Relationship made between the
        :func:`~geoh5py.data.data.Data.values` and elements of the
        :obj:`~geoh5py.shared.entity.Entity.parent` object.
        Association can be set from a :obj:`str` chosen from the list of available
        :obj:`~geoh5py.data.data_association_enum.DataAssociationEnum` options.
        """
        return self._association

    @association.setter
    def association(self, value: str | DataAssociationEnum):
        if isinstance(value, str):
            if value.upper() not in DataAssociationEnum.__members__:
                raise ValueError(
                    f"Association flag should be one of {DataAssociationEnum.__members__}"
                )

            value = getattr(DataAssociationEnum, value.upper())

        if not isinstance(value, DataAssociationEnum):
            raise TypeError(f"Association must be of type {DataAssociationEnum}")

        self._association = value

    @property
    def modifiable(self) -> bool:
        """
        :obj:`bool` Entity can be modified.
        """
        return self._modifiable

    @modifiable.setter
    def modifiable(self, value: bool):
        self._modifiable = value
        self.workspace.update_attribute(self, "attributes")

    @property
    def entity_type(self) -> DataType:
        """
        :obj:`~geoh5py.data.data_type.DataType`
        """
        return self._entity_type

    @entity_type.setter
    def entity_type(self, data_type: DataType):
        self._entity_type = data_type
        self.workspace.update_attribute(self, "entity_type")

    @classmethod
    @abstractmethod
    def primitive_type(cls) -> PrimitiveTypeEnum:
        ...

    def add_file(self, file: str):
        """
        Alias not implemented from base Entity class.
        """
        raise NotImplementedError("Data entity cannot contain files.")

    def remove_data_from_group(
        self, data: list | Entity | uuid.UUID | str, name: str | None = None
    ):
        """
        Remove self from a property group.
        """

    def mask_by_extent(
        self, extent: np.ndarray, inverse: bool = False
    ) -> np.ndarray | None:
        """
        Sub-class extension of :func:`~geoh5py.shared.entity.Entity.mask_by_extent`.

        Uses the parent object's vertices or centroids coordinates.
        """
        if self.association is DataAssociationEnum.VERTEX:
            return mask_by_extent(self.parent.vertices, extent, inverse=inverse)

        if self.association is DataAssociationEnum.CELL:
            if getattr(self.parent, "centroids", None) is not None:
                return mask_by_extent(self.parent.centroids, extent, inverse=inverse)

            indices = mask_by_extent(self.parent.vertices, extent, inverse=inverse)
            if indices is not None:
                indices = np.all(indices[self.parent.cells], axis=1)

            return indices

        return None

    def __call__(self):
        return self.values
