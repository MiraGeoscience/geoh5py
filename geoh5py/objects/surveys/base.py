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


# pylint: disable=no-member, too-many-lines, too-many-ancestors
# mypy: disable-error-code="attr-defined"

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from geoh5py.objects.object_base import ObjectBase
from geoh5py.shared.utils import copy_no_reference


if TYPE_CHECKING:
    from geoh5py.groups import Group
    from geoh5py.shared.entity import EntityContainer
    from geoh5py.workspace import Workspace


class BaseSurvey(ObjectBase, ABC):
    """
    A base survey object.
    """

    __INPUT_TYPE = None
    __TYPE = None
    __UNITS = None

    @property
    @abstractmethod
    def complement(self):
        """Returns the complement object for self."""

    def copy(  # pylint: disable=too-many-arguments
        self,
        parent: Group | Workspace | None = None,
        *,
        copy_children: bool = True,
        clear_cache: bool = False,
        mask: np.ndarray | None = None,
        **kwargs,
    ):
        """
        Sub-class extension of :func:`~geoh5py.objects.cell_object.CellObject.copy`.
        """
        if parent is None:
            parent = self.parent

        new_entity = super().copy(
            parent=parent,
            clear_cache=clear_cache,
            copy_children=copy_children,
            mask=mask,
            omit_list=self.omit_list,
            **kwargs,
        )

        if self.metadata is not None:
            new_entity.metadata = copy_no_reference(self.metadata)

        if self.complement is not None:
            self.copy_complement(
                new_entity,
                parent=parent,
                copy_children=copy_children,
                clear_cache=clear_cache,
                mask=mask,
            )

        return new_entity

    def copy_complement(
        self,
        new_entity,
        *,
        parent: Group | Workspace | None = None,
        copy_children: bool = True,
        clear_cache: bool = False,
        mask: np.ndarray | None = None,
    ):
        """
        Copy the complement entity to the new entity.

        :param new_entity: New entity to copy the complement to.
        :param parent: Parent group or workspace.
        :param copy_children: Copy children entities.
        :param clear_cache: Clear the cache.
        :param mask: Mask on vertices to apply to the data.
        """
        if self.complement is None:
            return None

        # Reset the mask based on Tx ID if it exists
        mask = new_entity.get_complement_mask(mask, self.complement)

        new_complement = self.complement._super_copy(  # pylint: disable=protected-access
            parent=parent,
            omit_list=self.omit_list,
            copy_children=copy_children,
            clear_cache=clear_cache,
            mask=mask,
        )

        setattr(
            new_entity,
            self.type_map[self.complement.type],
            new_complement,
        )

        setattr(
            new_complement,
            self.type_map[new_entity.type],
            new_entity,
        )

        new_entity.renumber_reference_ids()

        return new_complement

    @abstractmethod
    def get_complement_mask(self, mask: np.ndarray, complement) -> np.ndarray:
        """
        Get the complement mask based on the input mask.

        :param mask: Mask on the vertices.
        """

    @property
    @abstractmethod
    def omit_list(self) -> tuple:
        """List of attributes to skip on copy."""

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent: EntityContainer):
        self._set_parent(parent)

        if self.complement is not None:
            self.complement._set_parent(parent)  # pylint: disable=protected-access

    def renumber_reference_ids(self):
        """
        Renumber the reference IDs in the survey.
        """

    @property
    @abstractmethod
    def type(self):
        """Survey element type"""

    @property
    @abstractmethod
    def type_map(self) -> dict[str, str]:
        """Map of survey element types."""

    def _super_copy(
        self,
        parent: Group | Workspace | None = None,
        copy_children: bool = True,
        clear_cache: bool = False,
        mask: np.ndarray | None = None,
        **kwargs,
    ):
        """
        Call the super().copy of the class in copy_complement method.

        :return: New copy of the input entity.
        """
        return super().copy(
            parent=parent,
            copy_children=copy_children,
            clear_cache=clear_cache,
            mask=mask,
            **kwargs,
        )
