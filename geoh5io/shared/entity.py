import uuid
from abc import abstractmethod

from geoh5io.shared import EntityType


class Entity:
    def __init__(self, name: str, uid: uuid.UUID = None):
        self._uid = uid if uid is not None else uuid.uuid4()
        self._name = self.fix_up_name(name)
        # TODO: properties and setters
        self._visible = 1
        self._allow_delete = 1
        self._allow_rename = 1
        self._is_public = 1

    @property
    def uid(self) -> uuid.UUID:
        return self._uid

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, new_name: str):
        self._name = self.fix_up_name(new_name)

    @classmethod
    def fix_up_name(cls, name: str) -> str:
        """ If the given  name is not a valid one, transforms it to make it valid
        :return: a valid name built from the given name. It simply returns the given name
        if it was already valid.
        """
        # TODO: implement an actual fixup
        #  (possibly it has to be abstract with different implementations per Entity type)
        return name

    @abstractmethod
    def get_type(self) -> EntityType:
        ...
