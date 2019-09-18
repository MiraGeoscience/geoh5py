from typing import List

from . import interfaces


class DataHandler:
    def get_all(self,) -> List[interfaces.data.Data]:
        # TODO
        return []

    def find(self, query: interfaces.data.DataQuery) -> List[interfaces.data.Data]:
        # TODO
        pass

    def get(self, uid: interfaces.shared.Uuid) -> interfaces.data.Data:
        # TODO
        pass

    def get_float_values(
        self, data: interfaces.shared.Uuid, slab: interfaces.data.DataSlab
    ) -> List[float]:
        # TODO
        pass

    def get_integer_values(
        self, data: interfaces.shared.Uuid, slab: interfaces.data.DataSlab
    ) -> List[int]:
        # TODO
        pass

    def get_text_values(
        self, data: interfaces.shared.Uuid, slab: interfaces.data.DataSlab
    ) -> List[str]:
        # TODO
        pass

    def get_referenced_values(
        self, data: interfaces.shared.Uuid, slab: interfaces.data.DataSlab
    ) -> interfaces.data.ReferencedValues:
        # TODO
        pass

    def get_datetime_values(
        self, data: interfaces.shared.Uuid, slab: interfaces.data.DataSlab
    ) -> List[str]:
        # TODO
        pass

    def get_filename_values(
        self, data: interfaces.shared.Uuid, slab: interfaces.data.DataSlab
    ) -> List[str]:
        # TODO
        pass

    def get_file_content(self, data: interfaces.shared.Uuid, file_name: str) -> str:
        # TODO
        pass

    def get_blob_values(
        self, data: interfaces.shared.Uuid, slab: interfaces.data.DataSlab
    ) -> List[int]:
        # TODO
        pass

    def get_blob_element(self, data: interfaces.shared.Uuid, index: int) -> str:
        # TODO
        pass

    def get_all_types(self,) -> List[interfaces.data.DataType]:
        # TODO
        pass

    def find_types(
        self, query: interfaces.data.DataTypeQuery
    ) -> List[interfaces.data.DataType]:
        # TODO
        pass

    def get_type(self, uid: interfaces.shared.Uuid) -> interfaces.data.DataType:
        # TODO
        pass

    def set_public(
        self, entities: List[interfaces.shared.Uuid], is_public: bool
    ) -> None:
        # TODO
        pass

    def set_visible(
        self, entities: List[interfaces.shared.Uuid], visible: bool
    ) -> None:
        # TODO
        pass

    def set_allow_delete(
        self, entities: List[interfaces.shared.Uuid], allow: bool
    ) -> None:
        # TODO
        pass

    def set_allow_rename(
        self, entities: List[interfaces.shared.Uuid], allow: bool
    ) -> None:
        # TODO
        pass

    def rename(self, entities: interfaces.shared.Uuid, new_name: str) -> None:
        # TODO
        pass
