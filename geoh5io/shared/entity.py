import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Union

if TYPE_CHECKING:
    from geoh5io import shared


class Entity(ABC):

    _attribute_map = {
        "Allow delete": "allow_delete",
        "Allow move": "allow_move",
        "Allow rename": "allow_rename",
        "ID": "uid",
        "Name": "name",
        "Public": "public",
    }

    def __init__(self, **kwargs):

        self._uid: uuid.UUID = uuid.uuid4()
        self._name = "Entity"
        self._parent = None
        self._children: List = []
        self._visible = True
        self._allow_delete = True
        self._allow_move = False
        self._allow_rename = True
        self._public = True
        self._existing_h5_entity = False
        self._update_h5: List[str] = []

        for attr, item in kwargs.items():
            try:
                if attr in self._attribute_map.keys():
                    attr = self._attribute_map[attr]
                setattr(self, attr, item)
            except AttributeError:
                continue

    @property
    def attribute_map(self):
        return self._attribute_map

    @property
    def existing_h5_entity(self) -> bool:
        return self._existing_h5_entity

    @existing_h5_entity.setter
    def existing_h5_entity(self, value: bool):
        self._existing_h5_entity = value

    @property
    def update_h5(self):
        return self._update_h5

    @update_h5.setter
    def update_h5(self, values: Union[List, str]):

        if not isinstance(values, list):
            values = [values]

        # Check if re-setting the list or appending
        if len(values) == 0:
            self._update_h5 = []
        else:
            for value in values:
                if value not in self._update_h5:
                    self._update_h5.append(value)

    @property
    def uid(self) -> uuid.UUID:
        return self._uid

    @uid.setter
    def uid(self, uid: Union[str, uuid.UUID]):

        if isinstance(uid, str):
            uid = uuid.UUID(uid)

        self._uid = uid

    @uid.getter
    def uid(self):

        if self._uid is None:
            self._uid = uuid.uuid4()

        return self._uid

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, new_name: str):
        self._name = self.fix_up_name(new_name)
        self.update_h5 = "attributes"

    @property
    def visible(self) -> bool:
        return self._visible

    @visible.setter
    def visible(self, value: bool):
        self._visible = value
        self.update_h5 = "attributes"

    @property
    def allow_delete(self) -> bool:
        """
        bool: if True the Entity can be deleted from the geoh5
        """
        return self._allow_delete

    @allow_delete.setter
    def allow_delete(self, value: bool):
        self._allow_delete = value
        self.update_h5 = "attributes"

    @property
    def allow_move(self) -> bool:
        """
        bool: if True the Entity can change parent
        """
        return self._allow_move

    @allow_move.setter
    def allow_move(self, value: bool):
        self._allow_move = value
        self.update_h5 = "attributes"

    @property
    def allow_rename(self) -> bool:
        """
        bool: if True the Entity can change name
        """
        return self._allow_rename

    @allow_rename.setter
    def allow_rename(self, value: bool):
        self._allow_rename = value
        self.update_h5 = "attributes"

    @property
    def public(self) -> bool:
        """
        public

        bool: if True the Entity is visible in camera
        """
        return self._public

    @public.setter
    def public(self, value: bool):
        self._public = value
        self.update_h5 = "attributes"

    @property
    def workspace(self):
        """
        Workspace to which the Entity belongs
        """
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
        """
        Parental entity in the workspace tree
        """
        return self._parent

    @parent.setter
    def parent(self, parent: Union["shared.Entity", uuid.UUID]):

        if isinstance(parent, uuid.UUID):
            uid = parent
        else:
            uid = parent.uid

        self._parent = self.workspace.get_entity(uid)[0]
        self._parent.add_child(self)

    @parent.getter
    def parent(self):
        if (self._parent is None) and (self.name != "Workspace"):
            self._parent = self.workspace.get_entity("Workspace")[0]
            self._parent.add_child(self)

        return self._parent

    @property
    def children(self):
        """
        List of children entities in the workspace tree
        """
        return self._children

    def add_child(self, entity):
        """
        add_child

        Parameters
        ----------
        entity: Entity
            Add a child entity to the list of children
        """
        self._children.append(entity)

    @classmethod
    def create(cls, workspace, **kwargs):
        """
        Function to create an object with data

        :param workspace: Workspace to be added to
        :param kwargs: List of keyword arguments

        :return: Entity: Registered to the workspace.
        """

        if "entity_type_uid" in kwargs.keys():
            entity_type_kwargs = {"entity_type": {"uid": kwargs["entity_type_uid"]}}
        else:
            entity_type_kwargs = {}

        entity_kwargs = {"entity": kwargs}
        new_object = workspace.create_entity(
            cls, **{**entity_kwargs, **entity_type_kwargs}
        )

        # Add to root if parent is not set
        if new_object.parent is None:
            new_object.parent = workspace.root

        workspace.finalize()

        return new_object
