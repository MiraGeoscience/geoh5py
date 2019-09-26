from typing import TYPE_CHECKING, List

from geoh5io import interfaces
from geoh5io.workspace import Workspace

if TYPE_CHECKING:
    from geoh5io.interfaces.groups import Group as i_Group
    from geoh5io.interfaces.groups import GroupQuery as i_GroupQuery
    from geoh5io.interfaces.shared import Uuid as i_Uuid


class GroupsHandler:
    @staticmethod
    def get_root() -> i_Group:
        root = Workspace.active().root

        root_entity = interfaces.shared.Entity(
            uid=interfaces.shared.Uuid(str(root.uid)),
            type_uid=interfaces.shared.Uuid(str(root.entity_type.uid)),
            name=root.name,
            visible=root.visible,
            allow_delete=root.allow_delete,
            allow_rename=root.allow_rename,
            is_public=root.is_public,
        )

        return interfaces.groups.Group(entity_=root_entity, allow_move=root.allow_move)

    def get_type(self, group_class: int) -> i_Uuid:
        # TODO
        pass

    def get_class(self, type_uid: i_Uuid) -> int:
        # TODO
        pass

    @staticmethod
    def get_all() -> List[i_Group]:
        # TODO: get from workspace
        # return geoh5io.workspace.Workspace.instance().all_groups()
        return []

    def find(self, query: i_GroupQuery) -> List[i_Group]:
        # TODO
        pass

    def set_allow_move(self, groups: List[i_Uuid], allow: bool) -> None:
        # TODO
        pass

    def move_to_group(self, groups: List[i_Uuid], destination_group: i_Uuid) -> None:
        # TODO
        pass

    def create(self, type_uid: i_Uuid) -> i_Group:
        # TODO
        pass

    def set_public(self, entities: List[i_Uuid], is_public: bool) -> None:
        # TODO
        pass

    def set_visible(self, entities: List[i_Uuid], visible: bool) -> None:
        # TODO
        pass

    def set_allow_delete(self, entities: List[i_Uuid], allow: bool) -> None:
        # TODO
        pass

    def set_allow_rename(self, entities: List[i_Uuid], allow: bool) -> None:
        # TODO
        pass

    def rename(self, entities: i_Uuid, new_name: str) -> None:
        # TODO
        pass
