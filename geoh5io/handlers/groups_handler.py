from typing import List

from geoh5io import interfaces


class GroupsHandler:
    def get_root(self,) -> interfaces.groups.Group:
        # TODO
        pass

    def get_type(self, group_class: int) -> interfaces.shared.Uuid:
        # TODO
        pass

    def get_class(self, type_uid: interfaces.shared.Uuid) -> int:
        # TODO
        pass

    @staticmethod
    def get_all() -> List[interfaces.groups.Group]:
        # TODO: get from workspace
        # return geoh5io.workspace.Workspace.instance().all_groups()
        return []

    def find(
        self, query: interfaces.groups.GroupQuery
    ) -> List[interfaces.groups.Group]:
        # TODO
        pass

    def set_allow_move(self, groups: List[interfaces.shared.Uuid], allow: bool) -> None:
        # TODO
        pass

    def move_to_group(
        self,
        groups: List[interfaces.shared.Uuid],
        destination_group: interfaces.shared.Uuid,
    ) -> None:
        # TODO
        pass

    def create(self, type_uid: interfaces.shared.Uuid) -> interfaces.groups.Group:
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
