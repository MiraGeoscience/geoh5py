import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from geoh5io import shared


class Entity(ABC):
    def __init__(self, name: str, uid: uuid.UUID = None):
        if uid is not None:
            assert uid.int != 0
            self._uid = uid
        else:
            self._uid = uuid.uuid4()
        self._name = self.fix_up_name(name)
        self._parent = None
        self._children: List[Entity] = []
        self._visible = True
        self._allow_delete = True
        self._allow_rename = True
        self._public = True
        self._existing_h5_entity = False
        self._update_h5 = False

    @property
    def existing_h5_entity(self) -> bool:
        return self._existing_h5_entity

    @existing_h5_entity.setter
    def existing_h5_entity(self, value: bool):
        self._existing_h5_entity = value

    @property
    def update_h5(self) -> bool:
        return self._update_h5

    @update_h5.setter
    def update_h5(self, value: bool):
        self._update_h5 = value

    @property
    def uid(self) -> uuid.UUID:
        return self._uid

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, new_name: str):
        self._name = self.fix_up_name(new_name)

    @property
    def visible(self) -> bool:
        return self._visible

    @visible.setter
    def visible(self, value: bool):
        self._visible = value

    @property
    def allow_delete(self) -> bool:
        return self._allow_delete

    @allow_delete.setter
    def allow_delete(self, value: bool):
        self._allow_delete = value

    @property
    def allow_rename(self) -> bool:
        return self._allow_rename

    @allow_rename.setter
    def allow_rename(self, value: bool):
        self._allow_rename = value

    @property
    def is_public(self) -> bool:
        return self._public

    @is_public.setter
    def is_public(self, value: bool):
        self._public = value

    @property
    def public(self) -> bool:
        return self._public

    @public.setter
    def public(self, value: bool):
        self._public = value

    @property
    def workspace(self):
        return self.entity_type.workspace

    @classmethod
    def fix_up_name(cls, name: str) -> str:
        """ If the given  name is not a valid one, transforms it to make it valid
        :return: a valid name built from the given name. It simply returns the given name
        if it was already valid.
        """
        # TODO: implement an actual fixup
        #  (possibly it has to be abstract with different implementations per Entity type)
        return name

    @property
    @abstractmethod
    def entity_type(self) -> "shared.EntityType":
        ...

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent):
        """
        The parent of an object in the workspace
        :return: Entity: Parent entity
        """
        if parent:
            if isinstance(parent, uuid.UUID):
                uid = parent
            else:
                if isinstance(parent, Entity):
                    uid = parent.uid
                else:
                    uid = self.workspace.get_entity("Workspace")[0].uid

            # self.workspace.tree[self.uid]["parent"] = uid
            #
            # # Add as children of parent in tree
            # if self.uid not in self.workspace.tree[uid]["children"]:
            #     self.workspace.tree[uid]["children"] += [self.uid]

            self._parent = self.workspace.get_entity(uid)[0]
            self._parent.add_child(self)

        else:
            self._parent = None

    # def get_parent(self):
    #     """
    #     Function to fetch the parent of an object from the workspace tree
    #     :return: Entity: Parent entity of object
    #     """
    #     workspace = self.workspace
    #     return workspace.get_entity(workspace.tree[self.uid]["parent"])[0]

    @property
    def children(self):

        return self._children

    def add_child(self, entity):

        self._children.append(entity)
