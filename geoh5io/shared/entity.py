import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Union

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

    def __init__(self, name: str = "Entity", uid: uuid.UUID = None):
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
        self._allow_move = False
        self._allow_rename = True
        self._public = True
        self._existing_h5_entity = False
        self._update_h5: List[str] = []

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
    def parent(self, parent):
        if parent:
            if isinstance(parent, uuid.UUID):
                uid = parent
            else:
                if isinstance(parent, Entity):
                    uid = parent.uid
                else:
                    uid = self.workspace.get_entity("Workspace")[0].uid

            self._parent = self.workspace.get_entity(uid)[0]
            self._parent.add_child(self)

        else:
            self._parent = None

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
        object_type = cls.find_or_create_type(workspace)

        if object_type.name is None:
            object_type.name = cls.__name__

        if object_type.description is None:
            object_type.description = cls.__name__

        new_object = cls(object_type)

        # Replace all attributes given as kwargs
        for attr, item in kwargs.items():
            try:
                setattr(new_object, attr, item)
            except AttributeError:
                pass

        # Add parent-child relationship
        if "parent" in kwargs.keys():
            if isinstance(kwargs["parent"], uuid.UUID):
                parent = workspace.get_entity(kwargs["parent"])[0]
            else:
                assert isinstance(
                    kwargs["parent"], Entity
                ), "Given 'parent' argument must be of type uuid.UUID or 'Entity'"

                parent = kwargs["parent"]
        else:
            parent = workspace.root

        new_object.parent = parent

        return new_object

    @classmethod
    @abstractmethod
    def find_or_create_type(cls, workspace, uid: Optional[uuid.UUID] = None):
        ...
