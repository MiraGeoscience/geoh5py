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

from .notype import NoTypeGroup


class RootGroup(NoTypeGroup):
    """The Root group of a workspace."""

    _default_name = "Workspace"

    def __init__(
        self,
        allow_move=False,
        allow_delete=False,
        allow_rename=False,
        **kwargs,
    ):
        super().__init__(
            allow_move=allow_move,
            allow_delete=allow_delete,
            allow_rename=allow_rename,
            parent=self,
            **kwargs,
        )

    @property
    def parent(self):
        """
        Parental entity of root is always None
        """
        return None

    @parent.setter
    def parent(self, _):
        pass
