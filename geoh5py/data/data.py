# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2025 Mira Geoscience Ltd.                                     '
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

import logging
import uuid
from abc import abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from ..shared import Entity
from ..shared.utils import mask_by_extent
from .data_association_enum import DataAssociationEnum


if TYPE_CHECKING:
    from geoh5py.data.data_type import DataType, ReferenceDataType

logger = logging.getLogger(__name__)


class Data(Entity):
    """
    Base class for Data entities.

    :param association: Relationship made between the parent object and values.
    :param modifiable: Entity can be modified.
    :param visible: Entity is visible. Only a single data can be visible at a time.
    :param values: Data values.
    """

    _attribute_map = Entity._attribute_map.copy()
    _attribute_map.update({"Association": "association", "Modifiable": "modifiable"})

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
        *,
        clear_cache: bool = False,
        mask: np.ndarray | None = None,
        **kwargs,
    ):
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

        if (
            self.values is not None
            and mask is not None
            and self.association
            in (
                DataAssociationEnum.VERTEX,
                DataAssociationEnum.CELL,
            )
        ):
            if self.association is DataAssociationEnum.CELL:
                mask = self.parent.validate_cell_mask(mask)
                n_values = parent.n_cells
            else:
                mask = self.parent.validate_mask(mask)
                n_values = parent.n_vertices

            # Trimming values to the mask
            if n_values < self.values.shape[0]:
                values = self.values[mask]
            else:
                values = self.values.copy()
                values[~mask] = self.nan_value

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
        *,
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

    @classmethod
    def default_type_uid(cls) -> uuid.UUID | None:
        """Abstract method to return the default type uid for the class."""
        return None

    def format_length(self, values: np.ndarray) -> np.ndarray:
        """
        Check for possible mismatch between the length of values
        :param values: the values to check.
        :return: the values with the right length.
        """
        if self.n_values is None:
            return values

        if self.entity_type.primitive_type.name != "COLOUR":
            values = np.ravel(values)

        if len(values) < self.n_values:
            full_vector = np.empty(self.n_values, dtype=values.dtype)
            full_vector[:] = self.nan_value
            full_vector[: len(values)] = values

            return full_vector

        if (
            len(values) > self.n_values
            and self.association is not DataAssociationEnum.OBJECT
        ):
            logger.warning(
                "Input 'values' of shape (%s,) expected. Array of shape %s provided for data %s.",
                self.n_values,
                values.shape,
                self.name,
            )
            return values[: self.n_values]

        return values

    @property
    def formatted_values(self):
        """
        The data with the right format for Geoscience ANALYST.
        """
        return self.values

    @property
    def n_values(self) -> int | None:
        """
        Number of expected data values based on :obj:`~geoh5py.data.data.Data.association`
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
        Relationship made between the
        :func:`~geoh5py.data.data.Data.values` and elements of the
        :obj:`~geoh5py.shared.entity.Entity.parent` object.
        """
        return self._association

    @property
    def entity_type(self):
        """
        Type of data.
        """
        return self._entity_type

    @entity_type.setter
    def entity_type(self, data_type: DataType | ReferenceDataType):
        self._entity_type = self.validate_entity_type(data_type)

        if self.on_file:
            self.workspace.update_attribute(self, "entity_type")

    @property
    def modifiable(self) -> bool:
        """
        Entity can be modified within ANALYST.
        """
        return self._modifiable

    @modifiable.setter
    def modifiable(self, value: bool):
        self._modifiable = value

        if self.on_file:
            self.workspace.update_attribute(self, "attributes")

    def mask_by_extent(
        self, extent: np.ndarray, inverse: bool = False
    ) -> np.ndarray | None:
        """
        Sub-class extension of :func:`~geoh5py.shared.entity.Entity.mask_by_extent`.

        Uses the parent object's vertices or centroids coordinates.

        :param extent: Array or coordinate defining the lower and upper bounds of the extent.
        :param inverse: Keep the inverse (clip) of the extent selection.
        """
        if self.association is DataAssociationEnum.VERTEX and hasattr(
            self.parent, "vertices"
        ):
            return mask_by_extent(self.parent.vertices, extent, inverse=inverse)

        if self.association is DataAssociationEnum.CELL:
            if hasattr(self.parent, "vertices") and hasattr(self.parent, "_cells"):
                indices = mask_by_extent(self.parent.vertices, extent, inverse=inverse)
                if indices is not None:
                    indices = np.all(indices[self.parent.cells], axis=1)

                return indices

            if hasattr(self.parent, "centroids"):
                return mask_by_extent(self.parent.centroids, extent, inverse=inverse)

        return None

    def primitive_type(self) -> Enum:
        """Deprecated method to return the primitive type of the class."""
        logger.warning(
            "The method `primitive_type` will be deprecated. "
            "Use `entity_type.primitive_type` instead."
        )
        return self.entity_type.primitive_type

    def validate_entity_type(self, entity_type: DataType) -> DataType:
        """
        Validate the entity type.

        :param entity_type: Entity type to validate.
        """
        primitive_type = getattr(entity_type, "primitive_type", None)
        if not primitive_type or not issubclass(
            type(self), entity_type.primitive_type.value
        ):
            raise TypeError(
                f"Input 'entity_type' with primitive_type '{primitive_type}' "
                f"does not match the data type {type(self)}."
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

        To be deprecated along with the standalone Drillhole class in future version.

        :param values: Values to validate.
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
    def values(self, values):
        self._values = self.validate_values(values)

        if self.on_file:
            self.workspace.update_attribute(self, "values")

    @property
    def visible(self) -> bool:
        """
        Whether the data is visible in camera (checked in ANALYST object tree).
        """
        return bool(self._visible)

    @visible.setter
    def visible(self, value: bool):
        if value and self.on_file:
            for child in self.parent.children:
                if child is self:
                    continue
                child.visible = False

        self._visible = value

        if self.on_file:
            self.workspace.update_attribute(self, "attributes")

    def __call__(self):
        return self.values
