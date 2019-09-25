from typing import List, TYPE_CHECKING

from geoh5io.workspace import Workspace
from geoh5io import interfaces

if TYPE_CHECKING:
    from geoh5io.interfaces.data import Data as i_Data
    from geoh5io.interfaces.data import DataQuery as i_DataQuery
    from geoh5io.interfaces.data import DataType as i_DataType
    from geoh5io.interfaces.data import DataTypeQuery as i_DataTypeQuery
    from geoh5io.interfaces.data import ReferencedValues as i_ReferencedValues
    from geoh5io.interfaces.data import DataSlab as i_DataSlab
    from geoh5io.interfaces.shared import Uuid as i_Uuid

class DataHandler:
    @staticmethod
    def get_all() -> List[i_Data]:
        # TODO: get from workspace
        # return geoh5io.workspace.Workspace.instance().all_data()
        return []

    def find(self, query: i_DataQuery) -> List[i_Data]:
        # TODO
        pass

    def get(self, uid: i_Uuid) -> i_Data:
        Workspace.active().find_data()
        # TODO
        return interfaces.shared.Uuid()

    def get_float_values(
        self, data: i_Uuid, slab: i_DataSlab
    ) -> List[float]:
        # TODO
        pass

    def get_integer_values(
        self, data: i_Uuid, slab: i_DataSlab
    ) -> List[int]:
        # TODO
        pass

    def get_text_values(
        self, data: i_Uuid, slab: i_DataSlab
    ) -> List[str]:
        # TODO
        pass

    def get_referenced_values(
        self, data: i_Uuid, slab: i_DataSlab
    ) -> i_ReferencedValues:
        # TODO
        pass

    def get_datetime_values(
        self, data: i_Uuid, slab: i_DataSlab
    ) -> List[str]:
        # TODO
        pass

    def get_filename_values(
        self, data: i_Uuid, slab: i_DataSlab
    ) -> List[str]:
        # TODO
        pass

    def get_file_content(self, data: i_Uuid, file_name: str) -> str:
        # TODO
        pass

    def get_blob_values(
        self, data: i_Uuid, slab: i_DataSlab
    ) -> List[int]:
        # TODO
        pass

    def get_blob_element(self, data: i_Uuid, index: int) -> str:
        # TODO
        pass

    def get_all_types(self,) -> List[i_DataType]:
        # TODO
        pass

    def find_types(
        self, query: i_DataTypeQuery
    ) -> List[i_DataType]:
        # TODO
        pass

    def get_type(self, uid: i_Uuid) -> i_DataType:
        # TODO
        pass

    def set_public(
        self, entities: List[i_Uuid], is_public: bool
    ) -> None:
        # TODO
        pass

    def set_visible(
        self, entities: List[i_Uuid], visible: bool
    ) -> None:
        # TODO
        pass

    def set_allow_delete(
        self, entities: List[i_Uuid], allow: bool
    ) -> None:
        # TODO
        pass

    def set_allow_rename(
        self, entities: List[i_Uuid], allow: bool
    ) -> None:
        # TODO
        pass

    def rename(self, entities: i_Uuid, new_name: str) -> None:
        # TODO
        pass
