# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2025-2026 Mira Geoscience Ltd.                                '
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


# pylint: skip-file

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from .. import interfaces
from ..workspace import Workspace


if TYPE_CHECKING:
    from ..interfaces.data import Data as i_Data
    from ..interfaces.data import DataQuery as i_DataQuery
    from ..interfaces.data import DataSlab as i_DataSlab
    from ..interfaces.data import DataType as i_DataType
    from ..interfaces.data import DataTypeQuery as i_DataTypeQuery
    from ..interfaces.data import ReferencedValues as i_ReferencedValues
    from ..interfaces.shared import Uuid as i_Uuid


class DataHandler:
    @staticmethod
    def get_all() -> list[i_Data]:
        Workspace.active().data
        # TODO
        return []

    def find(self, query: i_DataQuery) -> list[i_Data]:
        # TODO
        pass

    def get(self, uid: i_Uuid) -> i_Data:
        Workspace.active().find_data(uuid.UUID(uid.id))
        # TODO
        return interfaces.data.Data()

    def get_float_values(self, data: i_Uuid, slab: i_DataSlab) -> list[float]:
        # TODO
        pass

    def get_integer_values(self, data: i_Uuid, slab: i_DataSlab) -> list[int]:
        # TODO
        pass

    def get_text_values(self, data: i_Uuid, slab: i_DataSlab) -> list[str]:
        # TODO
        pass

    def get_referenced_values(
        self, data: i_Uuid, slab: i_DataSlab
    ) -> i_ReferencedValues:
        # TODO
        pass

    def get_datetime_values(self, data: i_Uuid, slab: i_DataSlab) -> list[str]:
        # TODO
        pass

    def get_filename_values(self, data: i_Uuid, slab: i_DataSlab) -> list[str]:
        # TODO
        pass

    def get_file_content(self, data: i_Uuid, file_name: str) -> str:
        # TODO
        pass

    def get_blob_values(self, data: i_Uuid, slab: i_DataSlab) -> list[int]:
        # TODO
        pass

    def get_blob_element(self, data: i_Uuid, index: int) -> str:
        # TODO
        pass

    def get_all_types(
        self,
    ) -> list[i_DataType]:
        # TODO
        pass

    def find_types(self, query: i_DataTypeQuery) -> list[i_DataType]:
        # TODO
        pass

    def get_type(self, uid: i_Uuid) -> i_DataType:
        # TODO
        pass

    def set_public(self, entities: list[i_Uuid], is_public: bool) -> None:
        # TODO
        pass

    def set_visible(self, entities: list[i_Uuid], visible: bool) -> None:
        # TODO
        pass

    def set_allow_delete(self, entities: list[i_Uuid], allow: bool) -> None:
        # TODO
        pass

    def set_allow_rename(self, entities: list[i_Uuid], allow: bool) -> None:
        # TODO
        pass

    def rename(self, entities: i_Uuid, new_name: str) -> None:
        # TODO
        pass
